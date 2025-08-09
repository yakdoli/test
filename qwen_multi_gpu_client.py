"""
Qwen2.5-VL-7B-Instruct 다중 GPU 최적화 클라이언트
Flash Attention 2 + 전체 GPU 리소스 최대 활용
"""

import asyncio
import torch
import gc
import os
import psutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modules.utils.prompt_utils import build_syncfusion_prompt
from modules.utils.md_staging import save_page_markdown
import config


class MultiGPUResourceManager:
    """다중 GPU 리소스 관리 및 최적화"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.device_properties = {}
        self.device_memory = {}
        
        if self.gpu_available:
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                self.device_properties[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'compute_capability': f"{props.major}.{props.minor}"
                }
                
    def get_optimal_multi_gpu_config(self) -> Dict[str, Any]:
        """다중 GPU 최적 설정 반환"""
        if not self.gpu_available or self.device_count < 2:
            return self._get_single_gpu_config()
        
        # 전체 GPU 메모리 계산
        total_gpu_memory = sum(props['total_memory'] for props in self.device_properties.values())
        total_gpu_memory_gb = total_gpu_memory / (1024**3)
        
        # 시스템 RAM 확인
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / (1024**3)
        
        print(f"🔧 다중 GPU 시스템 리소스:")
        print(f"   GPU 개수: {self.device_count}")
        print(f"   총 GPU 메모리: {total_gpu_memory_gb:.1f}GB")
        print(f"   시스템 RAM: {system_memory_gb:.1f}GB ({system_memory.percent}% 사용중)")
        
        for i, props in self.device_properties.items():
            memory_gb = props['total_memory'] / (1024**3)
            print(f"   GPU {i}: {props['name']} ({memory_gb:.1f}GB)")
        
        # Qwen2.5-VL-7B는 약 14GB 필요 (FP16 기준)
        # 다중 GPU로 분산 로드
        if total_gpu_memory_gb >= 20:  # 여유있는 다중 GPU VRAM
            device_config = {
                "device_map": "auto",  # 자동 분산
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "max_memory": self._calculate_max_memory_per_gpu()
            }
            print("✅ 다중 GPU 자동 분산 로드 (최고 성능)")
        elif total_gpu_memory_gb >= 16:  # 최소 다중 GPU VRAM
            device_config = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "load_in_8bit": True,  # 8비트 양자화로 메모리 절약
                "max_memory": self._calculate_max_memory_per_gpu()
            }
            print("⚡ 다중 GPU 8비트 양자화 로드 (메모리 절약)")
        else:
            # 단일 GPU 폴백
            return self._get_single_gpu_config()
        
        return device_config
    
    def _get_single_gpu_config(self) -> Dict[str, Any]:
        """단일 GPU 설정"""
        if not self.gpu_available:
            return {"device_map": "cpu", "torch_dtype": torch.float32}
        
        gpu_memory_gb = self.device_properties[0]['total_memory'] / (1024**3)
        
        if gpu_memory_gb >= 16:
            return {
                "device_map": "cuda:0",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            }
        else:
            return {
                "device_map": "cuda:0",
                "torch_dtype": torch.float16,
                "load_in_8bit": True
            }
    
    def _calculate_max_memory_per_gpu(self) -> Dict[int, str]:
        """각 GPU의 최대 사용 메모리 계산"""
        max_memory = {}
        
        for i in range(self.device_count):
            # 각 GPU 메모리의 90% 사용 (시스템 여유분 확보)
            total_memory = self.device_properties[i]['total_memory']
            usable_memory = int(total_memory * 0.9)
            max_memory[i] = f"{usable_memory // (1024**3)}GB"
        
        return max_memory
    
    def cleanup_all_gpus(self):
        """모든 GPU 메모리 정리"""
        if self.gpu_available:
            for i in range(self.device_count):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()
        
    def get_multi_gpu_memory_usage(self) -> Dict[str, Any]:
        """다중 GPU 메모리 사용량 반환"""
        usage = {"system_memory": psutil.virtual_memory().percent}
        
        if self.gpu_available:
            for i in range(self.device_count):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                total = self.device_properties[i]['total_memory'] / (1024**3)
                
                usage[f'gpu_{i}'] = {
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'total_gb': total,
                    'utilization': (allocated / total) * 100
                }
        
        return usage


class OptimizedMultiGPUQwenClient:
    """최적화된 다중 GPU Qwen2.5-VL 클라이언트"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.resource_manager = MultiGPUResourceManager()
        self.device_config = None
        self.processing_lock = threading.Lock()  # 동시 처리 제어
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0,
            'gpu_memory_peaks': {}
        }
        
    async def initialize_model(self) -> bool:
        """다중 GPU 최적화된 모델 초기화"""
        print("🚀 다중 GPU 최적화 Qwen2.5-VL-7B-Instruct 모델 초기화 중...")
        
        try:
            # 메모리 정리
            self.resource_manager.cleanup_all_gpus()
            
            # 최적 설정 결정
            self.device_config = self.resource_manager.get_optimal_multi_gpu_config()
            
            # 환경 변수 설정 (성능 최적화)
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
            
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
            print("🧠 다중 GPU 모델 로드 중...")
            load_kwargs = {
                "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
                "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
                **self.device_config
            }
            
            # Flash Attention 2 설정
            if config.QWEN_USE_FLASH_ATTENTION and self.resource_manager.gpu_available:
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    print("⚡ Flash Attention 2 활성화 (메모리 최적화)")
                except Exception:
                    print("⚠️ Flash Attention 2 미지원 - 기본 attention 사용")
            
            # 다중 GPU 병렬 처리를 위한 추가 설정
            if self.resource_manager.device_count > 1:
                load_kwargs["low_cpu_mem_usage"] = True
                print(f"🔗 다중 GPU 병렬 로드 설정 ({self.resource_manager.device_count}개 GPU)")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
            
            # Evaluation 모드로 설정
            self.model.eval()
            
            # 모델 분산 정보 출력
            if hasattr(self.model, 'hf_device_map'):
                print("📊 모델 디바이스 분산:")
                for layer, device in self.model.hf_device_map.items():
                    print(f"   {layer}: {device}")
            
            # 초기 메모리 사용량 기록
            initial_memory = self.resource_manager.get_multi_gpu_memory_usage()
            print("💾 초기 GPU 메모리 사용량:")
            for gpu_id, gpu_info in initial_memory.items():
                if gpu_id.startswith('gpu_'):
                    print(f"   {gpu_id}: {gpu_info['allocated_gb']:.1f}GB/{gpu_info['total_gb']:.1f}GB "
                          f"({gpu_info['utilization']:.1f}%)")
            
            print("✅ 다중 GPU 모델 초기화 완료")
            return True
            
        except Exception as e:
            print(f"❌ 다중 GPU 모델 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_syncfusion_prompt_legacy(self) -> str:
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

    # NOTE: Deprecated local prompt function (replaced by modules.utils.prompt_utils.build_syncfusion_prompt)
    # def get_syncfusion_prompt(self, image_path: Path) -> str:
        """Syncfusion 특화 프롬프트 (동적 메타데이터 포함)
        - 미세조정 데이터셋/RAG 일관성 강화를 위해 동적 컨텍스트를 포함한 지시문
        - 컨텍스트/메타 정보 표준화, 섹션 스키마 고정, 언어/코드/표/리스트/링크 처리 규약 강화
        - OCR 정규화 규칙 포함 (띄어쓰기, 하이픈 줄바꿈, 특수문자 통합 등)
        """
        import re
        from datetime import datetime

        file_name = image_path.name
        parent_dir = image_path.parent.name
        stem = image_path.stem
        m = re.search(r"page[_-]?(\d+)", stem, re.IGNORECASE)
        page_number = m.group(1) if m else "unknown"
        iso_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        return f"""
You are a meticulous technical documentation OCR and structuring agent specialized in Syncfusion SDK manuals.
Your task is to convert the given documentation image into HIGH-FIDELITY Markdown that is suitable for LLM fine-tuning datasets and RAG retrieval.

CONTEXT VALUES (use EXACTLY in the metadata header):
- source: image
- domain: syncfusion-sdk
- task: pdf-ocr-to-markdown
- language: auto (keep original; do not translate)
- source_filename: {file_name}
- document_name: {parent_dir}
- page_number: {page_number}
- page_id: {parent_dir}#page_{page_number}
- timestamp: {iso_ts}
- fidelity: lossless

GLOBAL OUTPUT CONTRACT (MUST FOLLOW EXACTLY):
- Top-level must start with an HTML comment metadata block:
  <!--
  source: image
  domain: syncfusion-sdk
  task: pdf-ocr-to-markdown
  language: auto (keep original; do not translate)
  source_filename: {file_name}
  document_name: {parent_dir}
  page_number: {page_number}
  page_id: {parent_dir}#page_{page_number}
  timestamp: {iso_ts}
  fidelity: lossless
  -->
- After metadata, output the structured content only in Markdown. No extra explanations.
- Do not invent content. If text is cropped/unclear, include "[unclear]" and keep position.
- Preserve all text as-is except for OCR normalization rules below.

OCR NORMALIZATION RULES:
- Merge hyphenated line breaks: "inter-
  face" -> "interface" when it's the same token.
- Normalize multiple spaces to single spaces, but preserve indentation inside code blocks.
- Preserve Unicode punctuation and math symbols as-is.
- Keep list numbering as shown (don't renumber).
- Keep casing exactly; do not title-case or sentence-case.

STRUCTURE SCHEMA (ENFORCE):
# {{Page/Main Title}}

## Overview
- 1-3 bullets summarizing the page scope using only visible text.

## Content
- Reconstruct hierarchy (H2/H3/H4) exactly as in image.
- Tables: use GitHub-flavored Markdown. Keep column order, headers, alignment if visible.
- Lists: preserve nesting and markers (-, *, 1.) as-is.
- Callouts: map to blockquotes with labels (Note:, Warning:, Tip:).
- Figures/Captions: include as "Figure: ..." lines when present.

## API Reference (if applicable)
- Namespace, Class, Members (Methods/Properties/Events/Enums) in subsections.
- Parameters table: Name | Type | Description | Default | Required
- Returns: Type + description.
- Exceptions: bullet list.

## Code Examples (multi-language supported)
- Extract ALL code exactly. Use fenced blocks with language: ```csharp, ```vb, ```xml, ```xaml, ```js, ```css, ```ts, ```python.
- Keep full signatures, imports/usings, comments, region markers.
- Inline code in text should be wrapped with backticks.

## Cross References
- Add See also: bullet list of explicit links/texts present on the page. Do not fabricate.

## RAG Annotations
- At the end, add an HTML comment with tags and keywords derived ONLY from visible content:
  <!-- tags: [product, module, control, api, version?] keywords: [k1, k2, ...] -->

ADDITIONAL RULES:
- Units, versions, file paths, and identifiers must be preserved exactly.
- Do not reflow long lines inside code blocks.
- Preserve table cell line breaks using <br> if present.
- For cross-page references without URLs, keep the exact anchor text.

Output now in the specified format.
"""

    async def convert_single_image_optimized(self, image_path: Path, page_num: int) -> Optional[str]:
        """다중 GPU 최적화된 단일 이미지 변환"""
        with self.processing_lock:  # 동시 처리 제어
            start_time = datetime.now()
            self.stats['total_requests'] += 1
            
            try:
                # 이미지 로드 및 전처리
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": build_syncfusion_prompt(image_path)}
                        ]
                    }
                ]
                
                # qwen-vl-utils를 사용한 비전 정보 처리
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                # 입력 데이터 처리
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # GPU로 이동 (첫 번째 GPU 디바이스 사용)
                if self.resource_manager.gpu_available:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # 최적화된 생성 설정
                # 결정론적 생성 설정: 미세조정 데이터셋 일관성 보장
                generation_config = {
                    "max_new_tokens": 8192,
                    "do_sample": False,
                    "top_p": 0.9,
                    "use_cache": True,
                    "repetition_penalty": 1.05,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                # Flash Attention 최적화된 텍스트 생성
                with torch.no_grad():
                    # 메모리 효율성을 위한 gradient checkpointing 비활성화 (추론 시)
                    if hasattr(self.model, 'gradient_checkpointing_enable'):
                        self.model.gradient_checkpointing_disable()
                    
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
                
                # GPU 메모리 피크 기록
                if self.resource_manager.gpu_available:
                    current_memory = self.resource_manager.get_multi_gpu_memory_usage()
                    for gpu_id, gpu_info in current_memory.items():
                        if gpu_id.startswith('gpu_'):
                            peak_key = f"{gpu_id}_peak"
                            current_usage = gpu_info['allocated_gb']
                            self.stats['gpu_memory_peaks'][peak_key] = max(
                                self.stats['gpu_memory_peaks'].get(peak_key, 0),
                                current_usage
                            )
                
                # 처리 후 메모리 정리
                if self.resource_manager.gpu_available:
                    torch.cuda.empty_cache()
                
                return output_text
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                print(f"❌ 이미지 변환 실패 ({image_path.name}): {e}")
                return None
    
    async def convert_images_to_markdown_parallel_optimized(self, image_paths: List[Path]) -> str:
        """다중 GPU 최적화된 병렬 이미지 변환"""
        if not self.model:
            if not await self.initialize_model():
                return "모델 초기화 실패"
        
        total_images = len(image_paths)
        # GPU 메모리를 고려한 최적 청크 크기
        chunk_size = min(config.CHUNK_SIZE, 2 if self.resource_manager.device_count > 1 else 3)
        
        print(f"🚀 다중 GPU 최적화 병렬 처리 시작: {total_images}개 이미지")
        print(f"📦 청크 크기: {chunk_size} (다중 GPU 메모리 최적화)")
        print(f"⚡ Flash Attention 2: {'활성화' if config.QWEN_USE_FLASH_ATTENTION else '비활성화'}")
        
        results = {}
        failed_pages = []
        start_time = datetime.now()
        
        # 청크 단위로 처리
        for i in range(0, total_images, chunk_size):
            chunk = image_paths[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_images + chunk_size - 1) // chunk_size
            
            print(f"\n📦 청크 {chunk_num}/{total_chunks} 처리 중 ({len(chunk)}개 이미지)")
            
            # GPU 메모리 상태 모니터링
            memory_before = self.resource_manager.get_multi_gpu_memory_usage()
            
            # 청크 내 이미지들을 순차 처리 (안정성 우선)
            for j, image_path in enumerate(chunk):
                page_num = i + j + 1
                print(f"   🔄 페이지 {page_num}/{total_images} 변환 중...")
                
                markdown_text = await self.convert_single_image_optimized(image_path, page_num)
                
                if markdown_text and markdown_text.strip():
                    results[page_num] = markdown_text
                    print(f"   ✅ 페이지 {page_num} 완료 ({len(markdown_text)}자)")
                else:
                    results[page_num] = f"<!-- 페이지 {page_num} 변환 실패 -->"
                    failed_pages.append(page_num)
                    print(f"   ❌ 페이지 {page_num} 실패")

                # MD 스테이징 저장 (페이지 단위)
                try:
                    if markdown_text and markdown_text.strip():
                        save_page_markdown(
                            image_path,
                            markdown_text,
                            mode="scale_out",
                            prompt=build_syncfusion_prompt(image_path),
                            extra_meta={
                                "client": "OptimizedMultiGPUQwenClient"
                            },
                        )
                except Exception as _e:
                    print(f"   ⚠️ MD 스테이징 저장 경고: {str(_e)}")
            
            # 청크 처리 후 메모리 상태
            memory_after = self.resource_manager.get_multi_gpu_memory_usage()
            
            print(f"📊 청크 {chunk_num} GPU 메모리:")
            for gpu_id in [k for k in memory_after.keys() if k.startswith('gpu_')]:
                before = memory_before.get(gpu_id, {}).get('allocated_gb', 0)
                after = memory_after[gpu_id]['allocated_gb']
                total = memory_after[gpu_id]['total_gb']
                print(f"   {gpu_id}: {after:.1f}GB/{total:.1f}GB (Δ{after-before:+.1f}GB)")
            
            # 청크 간 메모리 정리
            if chunk_num < total_chunks:
                print(f"   🧹 GPU 메모리 정리 중...")
                self.resource_manager.cleanup_all_gpus()
                await asyncio.sleep(1)
        
        # 결과 조합
        markdown_content = []
        for page_num in sorted(results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(results[page_num])
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 최종 통계 출력
        print(f"\n📊 다중 GPU 최적화 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"  📦 처리된 청크: {total_chunks}개")
        print(f"  📈 처리량: {len(image_paths) / total_time:.2f} 페이지/초")
        print(f"  ✅ 성공: {self.stats['successful_requests']}/{self.stats['total_requests']}")
        print(f"  ❌ 실패: {self.stats['failed_requests']}/{self.stats['total_requests']}")
        
        if self.stats['successful_requests'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['successful_requests']
            print(f"  ⚡ 평균 응답 시간: {avg_time:.2f}초/페이지")
        
        # GPU 메모리 피크 출력
        if self.stats['gpu_memory_peaks']:
            print(f"  💾 GPU 메모리 피크:")
            for gpu_peak, peak_gb in self.stats['gpu_memory_peaks'].items():
                print(f"    {gpu_peak}: {peak_gb:.1f}GB")
        
        if failed_pages:
            print(f"  ⚠️ 실패한 페이지: {sorted(failed_pages)}")
        
        return "\n".join(markdown_content)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.stats.copy()
        stats.update({
            'mode': 'multi_gpu_optimized',
            'gpu_count': self.resource_manager.device_count,
            'flash_attention_enabled': config.QWEN_USE_FLASH_ATTENTION,
            'device_config': self.device_config
        })
        return stats
    
    def cleanup(self):
        """리소스 정리"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.processor:
            del self.processor
            self.processor = None
        
        self.resource_manager.cleanup_all_gpus()
        print("🗑️ 다중 GPU 모델 메모리 정리 완료")


async def main():
    """테스트 함수"""
    client = OptimizedMultiGPUQwenClient()
    
    print("🧪 다중 GPU 최적화 클라이언트 테스트")
    if await client.initialize_model():
        print("✅ 다중 GPU 모델 초기화 성공")
        
        stats = client.get_performance_stats()
        print(f"📊 설정: {stats['mode']}, GPU: {stats['gpu_count']}개, Flash Attention: {stats['flash_attention_enabled']}")
        
        client.cleanup()
    else:
        print("❌ 다중 GPU 모델 초기화 실패")


if __name__ == "__main__":
    asyncio.run(main())