"""
Qwen2.5-VL-7B-Instruct 직접 사용 클라이언트
GPU/CPU/RAM 리소스를 최적으로 활용하는 비동기 처리
"""

import asyncio
import torch
import gc
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import psutil
from tqdm.asyncio import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


class ResourceManager:
    """시스템 리소스 관리 및 모니터링"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_optimal_device_config(self) -> Dict[str, Any]:
        """최적 디바이스 설정 반환"""
        if not self.gpu_available:
            return {"device_map": "cpu", "torch_dtype": torch.float32}
        
        # GPU 메모리 확인
        total_gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                              for i in range(self.device_count))
        total_gpu_memory_gb = total_gpu_memory / (1024**3)
        
        # 시스템 RAM 확인
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / (1024**3)
        
        print(f"🔧 시스템 리소스 정보:")
        print(f"   GPU: {self.device_count}개, 총 메모리: {total_gpu_memory_gb:.1f}GB")
        print(f"   RAM: {system_memory_gb:.1f}GB ({system_memory.percent}% 사용중)")
        
        # Qwen2.5-VL-7B는 약 14GB VRAM 필요 (FP16 기준)
        if total_gpu_memory_gb >= 16:  # 여유있는 VRAM
            device_config = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            }
            print("✅ GPU 전용 로드 (최고 성능)")
        elif total_gpu_memory_gb >= 12:  # 최소 VRAM
            device_config = {
                "device_map": "auto", 
                "torch_dtype": torch.float16,
                "load_in_8bit": True  # 8비트 양자화
            }
            print("⚡ GPU 8비트 양자화 로드 (메모리 절약)")
        elif system_memory_gb >= 32:  # CPU 폴백
            device_config = {
                "device_map": "cpu",
                "torch_dtype": torch.float32
            }
            print("🖥️ CPU 로드 (GPU 메모리 부족)")
        else:
            raise RuntimeError("시스템 리소스가 부족합니다. 최소 32GB RAM 또는 12GB VRAM이 필요합니다.")
        
        return device_config
    
    def cleanup_memory(self):
        """메모리 정리"""
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        usage = {"system_memory": psutil.virtual_memory().percent}
        
        if self.gpu_available:
            for i in range(self.device_count):
                memory_info = torch.cuda.memory_stats(i)
                allocated = memory_info.get('allocated_bytes.all.current', 0) / (1024**3)
                cached = memory_info.get('reserved_bytes.all.current', 0) / (1024**3)
                usage[f'gpu_{i}_allocated'] = allocated
                usage[f'gpu_{i}_cached'] = cached
        
        return usage


class DirectQwenVLClient:
    """Qwen2.5-VL-7B-Instruct 직접 사용 클라이언트"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.resource_manager = ResourceManager()
        self.device_config = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0
        }
        
    async def initialize_model(self) -> bool:
        """모델 초기화"""
        print("🚀 Qwen2.5-VL-7B-Instruct 모델 초기화 중...")
        
        try:
            self.device_config = self.resource_manager.get_optimal_device_config()
            
            # 토크나이저 로드
            print("📝 토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.QWEN_MODEL_PATH,
                trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
            )
            
            # 프로세서 로드
            print("🎯 프로세서 로드 중...")
            self.processor = AutoProcessor.from_pretrained(
                config.QWEN_MODEL_PATH,
                trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
            )
            
            # 모델 로드
            print("🧠 모델 로드 중...")
            load_kwargs = {
                "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
                "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
                **self.device_config
            }
            
            # Flash Attention 2 지원 확인
            if config.QWEN_USE_FLASH_ATTENTION and self.resource_manager.gpu_available:
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    print("⚡ Flash Attention 2 활성화")
                except Exception:
                    print("⚠️ Flash Attention 2 미지원 - 기본 attention 사용")
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(**load_kwargs)
            
            # Evaluation 모드로 설정
            self.model.eval()
            
            print("✅ 모델 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")
            return False
    
    def get_syncfusion_prompt(self) -> str:
        """Syncfusion SDK 매뉴얼에 특화된 프롬프트 생성"""
        return """Convert this Syncfusion SDK documentation image to structured markdown format optimized for LLM fine-tuning and RAG applications.

CRITICAL REQUIREMENTS:

## Code Processing
- Extract ALL code snippets with proper language identification
- Preserve exact syntax, indentation, and formatting
- Use appropriate code blocks with language tags (```csharp, ```vb, ```xml, etc.)
- Maintain complete method signatures, parameter lists, and return types
- Include inline code elements using backticks for class names, properties, methods

## API Documentation Structure
- Identify and properly format: Classes, Namespaces, Methods, Properties, Events, Enums
- Use consistent heading hierarchy (# for main topics, ## for classes, ### for methods)
- Create clear parameter tables with: Name | Type | Description | Default Value
- Document return values with type and description
- Extract exception information if present

## Technical Content Enhancement
- Preserve all technical terminology exactly as written
- Maintain version-specific information and compatibility notes
- Include performance considerations and best practices
- Extract configuration settings and their valid values
- Document dependencies and required assemblies

## Structured Output Format
- Use descriptive headers that include class/namespace context
- Create linkable anchors for cross-references
- Format examples with clear "Example:" or "Usage:" headers
- Include "See Also" sections for related APIs
- Add metadata comments for categorization

## Content Completeness
- Extract ALL visible text without omission
- Preserve table structures with proper markdown formatting
- Maintain numbered/bulleted lists with correct nesting
- Include notes, warnings, and tips in appropriate callout format
- Capture image captions and figure references

## RAG Optimization
- Use semantic section breaks for better chunking
- Include contextual keywords for improved searchability
- Maintain hierarchical relationships between parent/child concepts
- Add implicit context where beneficial for standalone understanding

Focus on creating documentation that serves as high-quality training data for LLM fine-tuning while being immediately useful for RAG retrieval systems."""

    async def convert_image_to_markdown(self, image_path: Path) -> Optional[str]:
        """단일 이미지를 마크다운으로 변환"""
        if not self.model:
            if not await self.initialize_model():
                return None
        
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        try:
            # 이미지 로드 및 전처리
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": self.get_syncfusion_prompt()}
                    ]
                }
            ]
            
            # qwen-vl-utils를 사용한 비전 정보 처리
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
            
            # GPU 디바이스로 이동
            if self.resource_manager.gpu_available:
                inputs = inputs.to(self.model.device)
            
            # 생성 설정
            generation_config = {
                "max_new_tokens": 4000,
                "do_sample": False,
                "temperature": 0.1,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # 텍스트 생성
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)
                
            # 응답 텍스트 추출
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_requests'] += 1
            self.stats['total_processing_time'] += processing_time
            
            # 메모리 정리 (GPU 메모리 관리)
            if self.resource_manager.gpu_available:
                torch.cuda.empty_cache()
            
            return output_text
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            print(f"❌ 이미지 변환 실패 ({image_path.name}): {e}")
            return None
    
    async def convert_images_to_markdown_parallel(self, image_paths: List[Path]) -> str:
        """이미지들을 병렬로 마크다운 변환 (청크 단위)"""
        if not self.model:
            if not await self.initialize_model():
                return "모델 초기화 실패"
        
        total_images = len(image_paths)
        chunk_size = min(config.CHUNK_SIZE, 4)  # GPU 메모리 고려하여 청크 크기 제한
        
        print(f"🚀 Direct Qwen2.5-VL 병렬 처리 시작: {total_images}개 이미지")
        print(f"📦 청크 크기: {chunk_size} (GPU 메모리 최적화)")
        
        results = {}
        failed_pages = []
        
        # 청크 단위로 처리
        for i in range(0, total_images, chunk_size):
            chunk = image_paths[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_images + chunk_size - 1) // chunk_size
            
            print(f"\n📦 청크 {chunk_num}/{total_chunks} 처리 중 ({len(chunk)}개 이미지)")
            
            # 청크 내 이미지들을 순차 처리 (GPU 메모리 안정성)
            for j, image_path in enumerate(chunk):
                page_num = i + j + 1
                print(f"   🔄 페이지 {page_num}/{total_images} 변환 중...")
                
                markdown_text = await self.convert_image_to_markdown(image_path)
                
                if markdown_text and markdown_text.strip():
                    results[page_num] = markdown_text
                    print(f"   ✅ 페이지 {page_num} 완료 ({len(markdown_text)}자)")
                else:
                    results[page_num] = f"<!-- 페이지 {page_num} 변환 실패 -->"
                    failed_pages.append(page_num)
                    print(f"   ❌ 페이지 {page_num} 실패")
                
                # 메모리 모니터링
                memory_usage = self.resource_manager.get_memory_usage()
                if memory_usage['system_memory'] > 85:
                    print(f"   ⚠️ 메모리 사용량 높음: {memory_usage['system_memory']:.1f}%")
                    self.resource_manager.cleanup_memory()
            
            # 청크 간 짧은 휴식
            if chunk_num < total_chunks:
                await asyncio.sleep(1)
        
        # 결과를 페이지 순서대로 정렬하여 마크다운 생성
        markdown_content = []
        for page_num in sorted(results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(results[page_num])
        
        # 통계 출력
        success_count = len(results) - len(failed_pages)
        avg_time = (self.stats['total_processing_time'] / self.stats['successful_requests'] 
                   if self.stats['successful_requests'] > 0 else 0)
        
        print(f"\n📊 Direct Qwen2.5-VL 처리 완료:")
        print(f"  ✅ 성공: {success_count}/{total_images}")
        print(f"  ❌ 실패: {len(failed_pages)}")
        print(f"  ⚡ 평균 처리 시간: {avg_time:.1f}초/페이지")
        
        if failed_pages:
            print(f"  ⚠️ 실패한 페이지: {failed_pages}")
        
        return "\n".join(markdown_content)
    
    def cleanup(self):
        """모델 정리"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.processor:
            del self.processor
            self.processor = None
        
        self.resource_manager.cleanup_memory()
        print("🗑️ 모델 메모리 정리 완료")


async def main():
    """테스트 함수"""
    client = DirectQwenVLClient()
    
    if await client.initialize_model():
        print("✅ 모델 초기화 성공")
        memory_usage = client.resource_manager.get_memory_usage()
        print(f"💾 현재 메모리 사용량: {memory_usage}")
    else:
        print("❌ 모델 초기화 실패")


if __name__ == "__main__":
    asyncio.run(main())