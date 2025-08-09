"""
Qwen2.5-VL 최대 스케일 클라이언트
GPU 리소스 활용률 9-13%에서 최대 활용률로 확장
"""

import asyncio
import torch
import gc
import os
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import multiprocessing

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


class MaxScaleResourceManager:
    """최대 스케일 GPU 리소스 관리 - 활용률 극대화 (단일 GPU/멀티 워커)"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_count = multiprocessing.cpu_count()
        self.device_properties: Dict[int, Dict[str, Any]] = {}
        
        if self.gpu_available:
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                self.device_properties[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                }
    
    def get_max_scale_config(self) -> Dict[str, Any]:
        """최대 스케일 설정 - GPU 활용률 극대화"""
        if not self.gpu_available:
            raise RuntimeError("GPU가 필요합니다.")
        
        total_gpu_memory = sum(props['total_memory'] for props in self.device_properties.values())
        total_gpu_memory_gb = total_gpu_memory / (1024**3)
        
        print(f"🔧 최대 스케일 시스템 분석:")
        print(f"   GPU: {self.device_count}개")
        print(f"   총 GPU 메모리: {total_gpu_memory_gb:.1f}GB")
        print(f"   CPU 코어: {self.cpu_count}개")
        
        # GPU당 수용 가능한 워커 수 계산
        per_gpu_capacity = self._calculate_per_gpu_worker_capacity()
        max_parallel_instances = sum(per_gpu_capacity.values())
        
        # 전체 최대 워커 수 제한 적용
        if config.MAX_TOTAL_GPU_WORKERS is not None:
            max_parallel_instances = min(max_parallel_instances, config.MAX_TOTAL_GPU_WORKERS)
        
        config_settings = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "max_memory": self._calculate_aggressive_memory_allocation(),
            "low_cpu_mem_usage": False,
            "use_flash_attention_2": True,
            "max_parallel_instances": max_parallel_instances,
            "per_gpu_capacity": per_gpu_capacity,
            "batch_size": max(1, getattr(config, 'PER_WORKER_BATCH_SIZE', 1)),
            "concurrent_requests": max_parallel_instances  # 워커 수만큼 동시 요청
        }
        
        print(f"⚡ 최대 스케일 설정:")
        print(f"   GPU당 워커 수용력: {per_gpu_capacity}")
        print(f"   총 병렬 인스턴스(워커): {max_parallel_instances}개")
        print(f"   배치 크기(워커당): {config_settings['batch_size']}")
        print(f"   동시 요청 상한: {config_settings['concurrent_requests']}")
        
        return config_settings
    
    def _calculate_aggressive_memory_allocation(self) -> Dict[int, str]:
        """공격적 메모리 할당 - 95% 활용"""
        max_memory: Dict[int, str] = {}
        
        for i in range(self.device_count):
            total_memory = self.device_properties[i]['total_memory']
            # 95% 메모리 사용 (최대 활용)
            usable_memory = int(total_memory * 0.95)
            max_memory[i] = f"{usable_memory // (1024**3)}GB"
        
        return max_memory
    
    def _calculate_per_gpu_worker_capacity(self) -> Dict[int, int]:
        """GPU 메모리/설정 기반 GPU당 워커 수용력 계산"""
        capacity: Dict[int, int] = {}
        if not self.gpu_available:
            return capacity
        
        for i, props in self.device_properties.items():
            total_gb = props['total_memory'] / (1024**3)
            usable_gb = total_gb * float(getattr(config, 'GPU_MAX_MEMORY_FRACTION', 0.9))
            est_per_worker = float(getattr(config, 'WORKER_ESTIMATED_VRAM_GB', 16.0))
            # 최소 1개 보장
            max_by_mem = max(1, int(usable_gb // est_per_worker))
            if getattr(config, 'MAX_WORKERS_PER_GPU', None) is not None:
                max_by_mem = min(max_by_mem, int(config.MAX_WORKERS_PER_GPU))
            capacity[i] = max_by_mem
        return capacity
    
    def build_worker_plan(self) -> List[int]:
        """전체 워커 수에 대한 GPU 할당 계획 생성 (리스트: device_id들의 나열)"""
        per_gpu_capacity = self._calculate_per_gpu_worker_capacity()
        device_ids: List[int] = []
        # 라운드 로빈으로 GPU당 capacity만큼 추가
        more = True
        while more:
            more = False
            for gpu_id, cap in per_gpu_capacity.items():
                if sum(1 for d in device_ids if d == gpu_id) < cap:
                    device_ids.append(gpu_id)
                    more = True
                    # 전체 한도 체크
                    if getattr(config, 'MAX_TOTAL_GPU_WORKERS', None) is not None and len(device_ids) >= int(config.MAX_TOTAL_GPU_WORKERS):
                        return device_ids
        return device_ids


class MaxScaleQwenWorker:
    """최대 스케일 워커 - 단일 GPU 인스턴스 (GPU별 직결, 내부 마이크로배치 병렬)"""
    
    def __init__(self, worker_id: int, device_id: int, executor: ThreadPoolExecutor, tokenizer, processor):
        self.worker_id = worker_id
        self.device_id = device_id
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._lock = asyncio.Lock()  # generate 호출 중복 방지
        self.max_concurrency = max(1, int(getattr(config, 'PER_WORKER_CONCURRENCY', 1)))
        self.stats = {
            'processed_requests': 0,
            'total_time': 0,
            'device_utilization': []
        }
        self.executor = executor
        self.tokenizer = tokenizer
        self.processor = processor
        
    async def initialize(self) -> bool:
        """워커 초기화"""
        try:
            print(f"🔧 워커 {self.worker_id} (GPU {self.device_id}) 초기화 중...")
            
            # 모델 로딩을 스레드 풀에서 실행하여 비동기 이벤트 루프 블로킹 방지
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self._load_model_sync)
            
            print(f"✅ 워커 {self.worker_id} 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 워커 {self.worker_id} 초기화 실패: {e}")
            return False
    
    def _load_model_sync(self):
        """모델 로딩 (동기) - 스레드 풀에서 실행"""
        # Hugging Face 캐시 디렉토리 설정
        os.environ["HF_HOME"] = config.HF_CACHE_DIR
        
        load_kwargs = {
            "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
            "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
            "device_map": f"cuda:{self.device_id}",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": False, # 워커에서 모델 로드 시 메타 텐서 오류 방지
        }
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        self.model.to(f"cuda:{self.device_id}") # 모델을 명시적으로 GPU로 이동
        self.model.eval()
    
    async def process_image_batch(self, image_batch: List[Path]) -> List[str]:
        """이미지 배치 처리 - 워커 내부 마이크로배치/동시 처리 지원"""
        async with self._lock:  # generate 호출 중복 방지 (모델 공유)
            start_time = datetime.now()
            results: List[str] = [None] * len(image_batch)
            
            try:
                # 마이크로배치 단위로 나눔
                micro = max(1, self.max_concurrency)
                for i in range(0, len(image_batch), micro):
                    chunk_paths = image_batch[i:i+micro]
                    # 동시 전처리 + 개별 generate를 태스크로 올려도, 같은 모델 공유라 실제로는 순차 generate가 안전함
                    # 대신 여기서는 전처리를 병렬화하고, generate는 순차로 합쳐 소폭 concurrency를 제공
                    preprocess_tasks = [asyncio.create_task(self._prepare_inputs(p)) for p in chunk_paths]
                    prepared = await asyncio.gather(*preprocess_tasks, return_exceptions=True)
                    
                    # 순차 generate (VRAM 스파이크 방지)
                    for idx_in, prep in enumerate(prepared):
                        j = i + idx_in
                        if isinstance(prep, Exception) or prep is None:
                            results[j] = f"<!-- 전처리 실패: {chunk_paths[idx_in].name} -->"
                            continue
                        markdown = await self._generate_from_prepared(*prep)
                        results[j] = markdown if markdown else f"<!-- 처리 실패: {chunk_paths[idx_in].name} -->"
                        self.stats['processed_requests'] += 1
                
                processing_time = (datetime.now() - start_time).total_seconds()
                self.stats['total_time'] += processing_time
                
                # GPU 활용률 기록 (가용 시)
                try:
                    gpu_util = torch.cuda.utilization(self.device_id)
                    self.stats['device_utilization'].append(gpu_util)
                except Exception:
                    pass
                
                return results
                
            except Exception as e:
                print(f"❌ 워커 {self.worker_id} 배치 처리 오류: {e}")
                return [f"<!-- 배치 처리 실패: {e} -->" for _ in image_batch]
    
    async def _process_single_image(self, image_path: Path) -> Optional[str]:
        """단일 이미지 처리 (레거시 경로)"""
        try:
            prep = await self._prepare_inputs(image_path)
            if prep is None:
                return None
            return await self._generate_from_prepared(*prep)
        except Exception as e:
            print(f"❌ 이미지 처리 실패 ({image_path.name}): {e}")
            return None

    async def _prepare_inputs(self, image_path: Path):
        """전처리/입력 준비 (병렬 가능)"""
        try:
            loop = asyncio.get_running_loop()
            # CPU 바운드 전처리 작업을 스레드 풀에서 실행
            inputs_data = await loop.run_in_executor(
                self.executor,
                self._prepare_inputs_sync, # 동기 함수 호출
                image_path
            )
            
            # GPU로 이동 (비동기)
            inputs = {k: v.to(f"cuda:{self.device_id}") if hasattr(v, 'to') else v 
                      for k, v in inputs_data.items()}
            return inputs, image_path
        except Exception as e:
            print(f"⚠ 전처리 실패 ({image_path.name}): {e}")
            return None

    def _prepare_inputs_sync(self, image_path: Path):
        """전처리/입력 준비 (동기) - 스레드 풀에서 실행"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": self._get_optimized_prompt()}
            ]
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        return inputs

    async def _generate_from_prepared(self, inputs, image_path: Path) -> Optional[str]:
        """준비된 입력으로부터 generate 수행 (순차/락 하에 호출됨)"""
        try:
            generation_config = {
                "max_new_tokens": 8192,
                "do_sample": False,
                "top_p": 0.9,
                "use_cache": True,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            torch.cuda.empty_cache()
            return output_text
        except Exception as e:
            print(f"❌ generate 실패 ({image_path.name}): {e}")
            return None
    
    def _get_optimized_prompt(self) -> str:
        """최적화된 프롬프트 (간결하지만 효과적)"""
        return """Convert this technical documentation image to structured markdown format.

Requirements:
- Extract ALL code snippets with proper language tags (```csharp, ```xml, etc.)
- Preserve exact syntax and formatting
- Use clear heading hierarchy (# ## ###)
- Create parameter tables: Name | Type | Description
- Include all visible text and technical details
- Format examples with "Example:" headers
- Maintain numbered/bulleted lists
- Add metadata for categorization

Focus on accuracy and completeness for LLM training data."""
    
    def cleanup(self):
        """워커 정리"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.processor:
            del self.processor
            self.processor = None
        
        torch.cuda.empty_cache()


from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import multiprocessing

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


class MaxScaleQwenClient:
    """최대 스케일 Qwen 클라이언트 - GPU 활용률 극대화 (단일 GPU당 다중 워커)"""
    
    def __init__(self):
        self.resource_manager = MaxScaleResourceManager()
        self.workers: List[MaxScaleQwenWorker] = []
        self.config: Optional[Dict[str, Any]] = None
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.executor = ThreadPoolExecutor() # CPU 바운드 작업을 위한 스레드 풀
        self.stats = {
            'total_images_processed': 0,
            'total_processing_time': 0,
            'peak_gpu_utilization': {},
            'throughput_stats': []
        }

    async def _preload_model_to_cache(self):
        """모델을 로컬 캐시에 미리 다운로드 (단일 스레드)"""
        print("🚀 모델을 로컬 캐시에 미리 다운로드 중...")
        os.environ["HF_HOME"] = config.HF_CACHE_DIR
        
        # 토크나이저와 프로세서 로드 (다운로드 트리거)
        self.preloaded_tokenizer = AutoTokenizer.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        self.preloaded_processor = AutoProcessor.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 모델 로드 (다운로드 트리거)
        load_kwargs = {
            "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
            "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2" if config.QWEN_USE_FLASH_ATTENTION else "eager",
            "low_cpu_mem_usage": False,
            "use_safetensors": True,
        }
        Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        print("✅ 모델 사전 다운로드 완료.")

    async def initialize_max_scale_system(self) -> bool:
        """최대 스케일 시스템 초기화"""
        print("🚀 최대 스케일 Qwen 시스템 초기화 중...")
        
        try:
            # 모델 사전 다운로드 (단일 스레드)
            await self._preload_model_to_cache()

            self.config = self.resource_manager.get_max_scale_config()
            
            # 워커 생성 및 초기화
            max_workers = self.config['max_parallel_instances']
            per_gpu_capacity: Dict[int, int] = self.config.get('per_gpu_capacity', {})
            device_plan = self.resource_manager.build_worker_plan() if not per_gpu_capacity else [d for g, cap in per_gpu_capacity.items() for d in [g]*cap]
            if config.MAX_TOTAL_GPU_WORKERS is not None:
                device_plan = device_plan[:config.MAX_TOTAL_GPU_WORKERS]
            
            print(f"👥 {len(device_plan)}개 워커 생성 중 (계획: {device_plan})...")
            
            # 계획에 따라 GPU에 워커 분배 (각 워커는 단일 GPU 전용)
            worker_init_tasks = []
            temp_workers = []
            for i, device_id in enumerate(device_plan):
                worker = MaxScaleQwenWorker(worker_id=i, device_id=device_id, executor=self.executor, 
                                            tokenizer=self.preloaded_tokenizer, processor=self.preloaded_processor)
                temp_workers.append(worker)
                worker_init_tasks.append(worker.initialize())

            # 모든 워커 초기화를 병렬로 실행
            init_results = await asyncio.gather(*worker_init_tasks, return_exceptions=True)

            for i, result in enumerate(init_results):
                if isinstance(result, Exception):
                    print(f"❌ 워커 {temp_workers[i].worker_id} (GPU {temp_workers[i].device_id}) 초기화 실패: {result}")
                elif result:
                    self.workers.append(temp_workers[i])
                    print(f"✅ 워커 {temp_workers[i].worker_id} (GPU {temp_workers[i].device_id}) 준비 완료")
                else:
                    print(f"❌ 워커 {temp_workers[i].worker_id} (GPU {temp_workers[i].device_id}) 초기화 실패")
            
            if not self.workers:
                print("❌ 사용 가능한 워커가 없습니다.")
                return False
            
            print(f"🎯 최대 스케일 시스템 준비 완료: {len(self.workers)}개 워커")
            return True
            
        except Exception as e:
            print(f"❌ 최대 스케일 시스템 초기화 실패: {e}")
            return False
    
    async def convert_images_max_scale(self, image_paths: List[Path]) -> str:
        """최대 스케일 이미지 변환 - 최대 GPU 활용률"""
        if not self.workers:
            if not await self.initialize_max_scale_system():
                return "최대 스케일 시스템 초기화 실패"
        
        total_images = len(image_paths)
        batch_size = self.config['batch_size']
        
        print(f"🚀 최대 스케일 변환 시작:")
        print(f"   총 이미지: {total_images}개")
        print(f"   워커 수: {len(self.workers)}개")
        print(f"   배치 크기(워커당): {batch_size}")
        print(f"   예상 동시 처리량: {len(self.workers) * batch_size}개")
        
        start_time = datetime.now()
        
        # 이미지를 배치로 분할
        image_batches = [
            image_paths[i:i + batch_size] 
            for i in range(0, total_images, batch_size)
        ]
        
        print(f"📦 {len(image_batches)}개 배치 생성")
        
        # 작업 큐 생성 및 배치 추가
        task_queue = asyncio.Queue()
        for batch in image_batches:
            await task_queue.put(batch)
        
        # 결과를 저장할 리스트 (순서 유지를 위해 인덱스 사용)
        results_map: Dict[int, List[str]] = {}
        results_lock = asyncio.Lock() # 결과 맵 접근 보호
        
        # 워커 코루틴 정의
        async def worker_task(worker: MaxScaleQwenWorker):
            nonlocal results_map, results_lock
            while True:
                batch_to_process = await task_queue.get()
                if batch_to_process is None: # Sentinel value to stop worker
                    task_queue.task_done()
                    break
                
                try:
                    # 배치 처리
                    batch_results = await worker.process_image_batch(batch_to_process)
                    
                    # 결과 맵에 저장 (원래 순서 유지를 위해)
                    async with results_lock:
                        # Find the original index of the first image in the batch
                        first_image_path = batch_to_process[0]
                        original_index = image_paths.index(first_image_path)
                        results_map[original_index] = batch_results
                        
                except Exception as e:
                    print(f"❌ 워커 {worker.worker_id} 배치 처리 중 오류 발생: {e}")
                    async with results_lock:
                        first_image_path = batch_to_process[0]
                        original_index = image_paths.index(first_image_path)
                        results_map[original_index] = [f"<!-- 처리 실패: {str(e)} -->" for _ in batch_to_process]
                finally:
                    task_queue.task_done()

        # 모든 워커에 대한 태스크 생성
        worker_coroutines = [worker_task(worker) for worker in self.workers]
        
        # 모든 워커가 작업을 마칠 때까지 기다림
        await asyncio.gather(*worker_coroutines)
        
        # 모든 작업이 완료되었음을 큐에 알림 (워커 종료용)
        for _ in self.workers:
            await task_queue.put(None)
        
        await task_queue.join() # 모든 태스크가 완료될 때까지 기다림
        
        # 결과 조합 (원래 이미지 순서대로)
        markdown_content = []
        page_counter = 1
        
        # results_map의 키(original_index)를 정렬하여 순서대로 결과 가져오기
        sorted_indices = sorted(results_map.keys())
        
        for original_index in sorted_indices:
            batch_results = results_map[original_index]
            for result in batch_results:
                if page_counter > 1:
                    markdown_content.append("\n---\n")
                markdown_content.append(f"<!-- 페이지 {page_counter} -->\n")
                markdown_content.append(result)
                page_counter += 1
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 통계 수집
        self._collect_performance_stats(total_time, total_images)
        
        print(f"\n📊 최대 스케일 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초")
        print(f"  🚀 처리량: {total_images / total_time:.2f} 페이지/초")
        print(f"  📈 워커당 평균: {total_images / len(self.workers) / total_time:.2f} 페이지/초")
        
        # GPU 활용률 통계
        self._print_gpu_utilization_stats()
        
        return "\n".join(markdown_content)
    
    def _collect_performance_stats(self, total_time: float, total_images: int):
        """성능 통계 수집"""
        self.stats['total_images_processed'] = total_images
        self.stats['total_processing_time'] = total_time
        self.stats['throughput_stats'].append({
            'images': total_images,
            'time': total_time,
            'throughput': total_images / total_time,
            'workers': len(self.workers)
        })
        
        # 워커별 GPU 활용률 수집
        for worker in self.workers:
            if worker.stats['device_utilization']:
                device_key = f"gpu_{worker.device_id}"
                max_util = max(worker.stats['device_utilization'])
                avg_util = sum(worker.stats['device_utilization']) / len(worker.stats['device_utilization'])
                
                if device_key not in self.stats['peak_gpu_utilization']:
                    self.stats['peak_gpu_utilization'][device_key] = {
                        'peak': max_util,
                        'average': avg_util,
                        'workers': []
                    }
                
                self.stats['peak_gpu_utilization'][device_key]['peak'] = max(
                    self.stats['peak_gpu_utilization'][device_key]['peak'], max_util
                )
                self.stats['peak_gpu_utilization'][device_key]['workers'].append(worker.worker_id)
    
    def _print_gpu_utilization_stats(self):
        """GPU 활용률 통계 출력"""
        print(f"\n💾 GPU 활용률 통계:")
        for device_key, stats in self.stats['peak_gpu_utilization'].items():
            print(f"  {device_key}:")
            print(f"    피크: {stats['peak']:.1f}%")
            print(f"    평균: {stats['average']:.1f}%")
            print(f"    워커: {len(stats['workers'])}개")
    
    def get_max_scale_stats(self) -> Dict[str, Any]:
        """최대 스케일 통계 반환"""
        return {
            'mode': 'max_scale_optimized',
            'worker_count': len(self.workers),
            'gpu_count': self.resource_manager.device_count,
            'max_parallel_instances': self.config.get('max_parallel_instances', 0) if self.config else 0,
            'batch_size': self.config.get('batch_size', 0) if self.config else 0,
            'performance_stats': self.stats
        }
    
    def cleanup_all_workers(self):
        """모든 워커 정리"""
        print("🧹 최대 스케일 시스템 정리 중...")
        
        for worker in self.workers:
            worker.cleanup()
        
        self.workers.clear()
        
        if self.resource_manager.gpu_available:
            for i in range(self.resource_manager.device_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        
        gc.collect()
        
        if self.executor:
            self.executor.shutdown()
            
        print("✅ 최대 스케일 시스템 정리 완료")


async def main():
    """테스트 함수"""
    client = MaxScaleQwenClient()
    
    print("🧪 최대 스케일 클라이언트 테스트")
    if await client.initialize_max_scale_system():
        print("✅ 최대 스케일 시스템 초기화 성공")
        
        stats = client.get_max_scale_stats()
        print(f"📊 시스템 설정:")
        print(f"   워커: {stats['worker_count']}개")
        print(f"   GPU: {stats['gpu_count']}개")
        print(f"   최대 병렬: {stats['max_parallel_instances']}")
        
        client.cleanup_all_workers()
    else:
        print("❌ 최대 스케일 시스템 초기화 실패")


if __name__ == "__main__":
    asyncio.run(main())