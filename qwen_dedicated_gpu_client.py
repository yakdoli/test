"""
GPU 전용 워커 시스템 - 각 워커는 단일 GPU 전용 사용
GPU간 오프로드 오버헤드 최소화를 위한 최적화
"""

import asyncio
import torch
import gc
import os
import psutil
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pickle

# CUDA 멀티프로세싱을 위한 spawn 방식 설정
multiprocessing.set_start_method('spawn', force=True)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


class DedicatedGPUResourceManager:
    """GPU 전용 리소스 관리자 - 오프로드 최소화"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_count = multiprocessing.cpu_count()
        self.device_properties = {}
        
        if self.gpu_available:
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                self.device_properties[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'compute_capability': f"{props.major}.{props.minor}",
                }
    
    def get_dedicated_gpu_config(self, device_id: int) -> Dict[str, Any]:
        """특정 GPU 전용 설정 반환"""
        if not self.gpu_available or device_id >= self.device_count:
            raise ValueError(f"GPU {device_id}는 사용할 수 없습니다.")
        
        device_memory = self.device_properties[device_id]['total_memory']
        device_memory_gb = device_memory / (1024**3)
        
        print(f"🔧 GPU {device_id} 전용 설정:")
        print(f"   디바이스: {self.device_properties[device_id]['name']}")
        print(f"   메모리: {device_memory_gb:.1f}GB")
        
        # 단일 GPU 전용 최적화 설정
        config_settings = {
            "device_map": f"cuda:{device_id}",  # 특정 GPU만 사용
            "torch_dtype": torch.bfloat16,
            "use_flash_attention_2": True,
            "low_cpu_mem_usage": True,
            # GPU간 오프로드 완전 차단
            "offload_folder": None,
            "offload_state_dict": False,
            # 메모리 최적화 (95% 사용)
            "max_memory": {device_id: f"{int(device_memory_gb * 0.95)}GB"}
        }
        
        return config_settings


def initialize_dedicated_worker(device_id: int, worker_config: Dict[str, Any]) -> bool:
    """프로세스별 GPU 전용 워커 초기화"""
    try:
        # CUDA 디바이스 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        torch.cuda.set_device(0)  # 보이는 첫 번째 디바이스 사용
        
        print(f"🔧 프로세스별 워커 {device_id} 초기화 중...")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 프로세서 로드
        processor = AutoProcessor.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 모델 로드 (단일 GPU 전용)
        load_kwargs = {
            "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
            "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
            "device_map": "cuda:0",  # 현재 프로세스의 유일한 GPU
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True
        }
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        model.eval()
        
        # 글로벌 변수로 저장
        globals()['model'] = model
        globals()['tokenizer'] = tokenizer
        globals()['processor'] = processor
        globals()['device_id'] = device_id
        
        print(f"✅ 프로세스별 워커 {device_id} 초기화 완료")
        return True
        
    except Exception as e:
        print(f"❌ 프로세스별 워커 {device_id} 초기화 실패: {e}")
        return False


def process_image_batch_dedicated(image_batch_data: bytes) -> List[str]:
    """GPU 전용 이미지 배치 처리"""
    try:
        # 직렬화된 데이터 복원
        image_paths = pickle.loads(image_batch_data)
        
        model = globals()['model']
        tokenizer = globals()['tokenizer'] 
        processor = globals()['processor']
        device_id = globals()['device_id']
        
        results = []
        
        for image_path in image_paths:
            try:
                # Syncfusion 특화 프롬프트 (간소화)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": get_optimized_prompt()}
                    ]
                }]
                
                # 전처리
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # 현재 GPU로 이동 (오프로드 없음)
                inputs = {k: v.to("cuda:0") if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
                
                # 최적화된 생성 설정
                generation_config = {
                    "max_new_tokens": 2500,
                    "do_sample": False,
                    "temperature": 0.1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                
                # 텍스트 생성 (단일 GPU에서만)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generation_config)
                
                # 응답 추출
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                results.append(output_text)
                
                # 즉시 메모리 정리 (오버헤드 최소화)
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ 이미지 처리 실패 ({image_path}): {e}")
                results.append(f"<!-- GPU {device_id} 처리 실패: {image_path.name} -->")
        
        return results
        
    except Exception as e:
        print(f"❌ 배치 처리 실패: {e}")
        return [f"<!-- 배치 처리 실패: {e} -->"]


def get_optimized_prompt() -> str:
    """최적화된 프롬프트 (빠른 처리용)"""
    return """Convert this technical documentation to markdown format.

Key requirements:
- Extract code snippets with proper language tags (```csharp, ```xml)
- Use clear heading structure (# ## ###)
- Include all visible text and technical details
- Format tables and lists properly
- Add code examples with proper formatting

Focus on accuracy and completeness."""


class DedicatedGPUQwenClient:
    """GPU 전용 Qwen 클라이언트 - 오프로드 최소화"""
    
    def __init__(self):
        self.resource_manager = DedicatedGPUResourceManager()
        self.process_pool = None
        self.workers_initialized = False
        self.stats = {
            'total_images_processed': 0,
            'total_processing_time': 0,
            'gpu_utilization_stats': {},
            'processing_times': []
        }
        
    async def initialize_dedicated_system(self) -> bool:
        """GPU 전용 시스템 초기화"""
        print("🚀 GPU 전용 시스템 초기화 중...")
        
        if not self.resource_manager.gpu_available:
            print("❌ GPU가 필요합니다.")
            return False
        
        gpu_count = self.resource_manager.device_count
        print(f"💾 사용 가능한 GPU: {gpu_count}개")
        
        try:
            # 각 GPU별 프로세스 풀 생성 (spawn 방식)
            ctx = multiprocessing.get_context('spawn')
            self.process_pool = ProcessPoolExecutor(
                max_workers=gpu_count,
                mp_context=ctx
            )
            
            print(f"✅ GPU 전용 시스템 초기화 완료: {gpu_count}개 프로세스")
            self.workers_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ GPU 전용 시스템 초기화 실패: {e}")
            return False
    
    async def convert_images_dedicated_gpu(self, image_paths: List[Path]) -> str:
        """GPU 전용 이미지 변환"""
        if not self.workers_initialized:
            if not await self.initialize_dedicated_system():
                return "GPU 전용 시스템 초기화 실패"
        
        total_images = len(image_paths)
        gpu_count = self.resource_manager.device_count
        
        # GPU별 배치 분할 (각 GPU가 전용으로 처리)
        images_per_gpu = total_images // gpu_count
        remainder = total_images % gpu_count
        
        batches = []
        start_idx = 0
        
        for i in range(gpu_count):
            batch_size = images_per_gpu + (1 if i < remainder else 0)
            if batch_size > 0:
                batch = image_paths[start_idx:start_idx + batch_size]
                batches.append(batch)
                start_idx += batch_size
        
        print(f"🚀 GPU 전용 변환 시작:")
        print(f"   총 이미지: {total_images}개")
        print(f"   GPU 수: {gpu_count}개")
        print(f"   배치 수: {len(batches)}개")
        print(f"   GPU별 평균: {total_images/gpu_count:.1f}개")
        
        start_time = datetime.now()
        
        # 각 배치를 별도 프로세스에서 처리 (GPU 전용)
        from qwen_dedicated_worker import process_image_batch_worker
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        for i, batch in enumerate(batches):
            # 배치 데이터 직렬화 (프로세스간 전송)
            batch_data = pickle.dumps(batch)
            device_id = i % gpu_count  # GPU 할당
            
            # 각 프로세스가 전용 GPU 사용
            task = loop.run_in_executor(
                self.process_pool, 
                process_image_batch_worker, 
                (device_id, batch_data)
            )
            tasks.append((task, device_id, len(batch)))
        
        print(f"⚡ {len(tasks)}개 GPU 전용 프로세스 실행 중...")
        
        # 모든 작업 완료 대기
        all_results = {}
        page_counter = 1
        
        for task, device_id, batch_size in tasks:
            try:
                batch_results = await task
                
                for result in batch_results:
                    all_results[page_counter] = result
                    page_counter += 1
                
                print(f"✅ GPU {device_id} 전용 처리 완료 ({batch_size}개)")
                
            except Exception as e:
                print(f"❌ GPU {device_id} 처리 실패: {e}")
                for _ in range(batch_size):
                    all_results[page_counter] = f"<!-- GPU {device_id} 처리 실패 -->"
                    page_counter += 1
        
        # 결과 조합
        markdown_content = []
        for page_num in sorted(all_results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(all_results[page_num])
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 통계 수집
        self._collect_performance_stats(total_time, total_images, gpu_count)
        
        print(f"\n📊 GPU 전용 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초")
        print(f"  🚀 처리량: {total_images / total_time:.2f} 페이지/초")
        print(f"  📈 GPU별 평균: {total_images / gpu_count / total_time:.2f} 페이지/초")
        print(f"  ⚡ 오프로드 오버헤드: 최소화됨 (GPU 전용)")
        
        return "\n".join(markdown_content)
    
    def _collect_performance_stats(self, total_time: float, total_images: int, gpu_count: int):
        """성능 통계 수집"""
        self.stats['total_images_processed'] = total_images
        self.stats['total_processing_time'] = total_time
        self.stats['processing_times'].append(total_time)
        self.stats['gpu_utilization_stats'] = {
            'gpu_count': gpu_count,
            'throughput_per_gpu': total_images / gpu_count / total_time,
            'total_throughput': total_images / total_time,
            'overhead_minimized': True
        }
    
    def get_dedicated_gpu_stats(self) -> Dict[str, Any]:
        """GPU 전용 통계 반환"""
        return {
            'mode': 'dedicated_gpu_optimized',
            'gpu_count': self.resource_manager.device_count,
            'offload_overhead': 'minimized',
            'worker_type': 'dedicated_single_gpu',
            'performance_stats': self.stats
        }
    
    def cleanup(self):
        """GPU 전용 시스템 정리"""
        print("🧹 GPU 전용 시스템 정리 중...")
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        self.workers_initialized = False
        
        # 모든 GPU 메모리 정리
        if self.resource_manager.gpu_available:
            for i in range(self.resource_manager.device_count):
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                except:
                    pass
        
        gc.collect()
        print("✅ GPU 전용 시스템 정리 완료")


async def main():
    """테스트 함수"""
    client = DedicatedGPUQwenClient()
    
    print("🧪 GPU 전용 클라이언트 테스트")
    if await client.initialize_dedicated_system():
        print("✅ GPU 전용 시스템 초기화 성공")
        
        stats = client.get_dedicated_gpu_stats()
        print(f"📊 시스템 설정:")
        print(f"   GPU 수: {stats['gpu_count']}개")
        print(f"   모드: {stats['mode']}")
        print(f"   오프로드: {stats['offload_overhead']}")
        
        client.cleanup()
    else:
        print("❌ GPU 전용 시스템 초기화 실패")


if __name__ == "__main__":
    asyncio.run(main())