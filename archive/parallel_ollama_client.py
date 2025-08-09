"""
Xinference API를 사용하는 Chunk 기반 배치 처리 클라이언트
메모리 효율성을 위해 chunk 단위로 순차 처리하되, chunk 내부는 병렬 처리
"""

import asyncio
import aiohttp
import json
import base64
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import config

class ChunkedAsyncXinferenceClient:
    def __init__(self):
        self.base_url = config.XINFERENCE_BASE_URL
        self.model_name = config.XINFERENCE_MODEL_NAME
        self.model_uid = config.XINFERENCE_MODEL_UID
        self.max_concurrent = config.MAX_CONCURRENT_REQUESTS
        self.request_timeout = config.REQUEST_TIMEOUT
        self.retry_delay = config.RETRY_DELAY
        self.chunk_size = config.CHUNK_SIZE
        
        # 동시 요청 수 제한을 위한 세마포어
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.stats_lock = asyncio.Lock()
        self.stats = {}
        
        self.reset_stats()

    def reset_stats(self):
        """통계 데이터를 초기화"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0,
            'concurrent_requests': 0
        }

    def encode_image_to_base64(self, image_path: Path) -> str:
        """이미지 파일을 base64로 인코딩"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def check_xinference_connection(self) -> bool:
        """Xinference 서버 연결 상태를 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception as e:
            print(f"❌ Xinference 연결 실패: {e}")
            return False

    async def check_model_availability(self) -> bool:
        """지정된 모델이 사용 가능한지 확인하고 model_uid 설정"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('data', [])
                        for model in models:
                            if model.get('id', '').startswith(self.model_name):
                                self.model_uid = model.get('id')
                                return True
                    return False
        except Exception as e:
            print(f"❌ 모델 확인 실패: {e}")
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

    async def convert_single_image(self, session: aiohttp.ClientSession, image_path: Path, page_num: int) -> Tuple[int, Optional[str]]:
        """
        단일 이미지를 마크다운으로 변환 (Xinference API 사용)
        """
        async with self.semaphore:
            async with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['concurrent_requests'] += 1
            
            start_time = time.time()
            
            try:
                image_base64 = self.encode_image_to_base64(image_path)
                
                # Xinference OpenAI 호환 API 사용
                payload = {
                    "model": self.model_uid or self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.get_syncfusion_prompt()
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4000,
                    "stream": False
                }
                
                max_retries = 3
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                        
                        async with session.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                            timeout=timeout,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            
                            if response.status == 200:
                                result = await response.json()
                                markdown_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                                
                                if markdown_text.strip():  # 빈 내용 검사
                                    async with self.stats_lock:
                                        self.stats['successful_requests'] += 1
                                        self.stats['total_time'] += time.time() - start_time
                                    return page_num, markdown_text
                                else:
                                    last_exception = "Empty response received"
                            else:
                                response_text = await response.text()
                                error_msg = f"HTTP {response.status}: {response_text[:200]}"
                                last_exception = error_msg
                                
                    except asyncio.TimeoutError as e:
                        error_msg = f"Timeout after {self.request_timeout}s"
                        last_exception = error_msg
                    except aiohttp.ClientError as e:
                        error_msg = f"Network error: {str(e)}"
                        last_exception = error_msg
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        last_exception = error_msg
                    
                    # 재시도 대기 (마지막 시도가 아닌 경우)
                    if attempt < max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        await asyncio.sleep(wait_time)
                
                # 모든 재시도 실패
                async with self.stats_lock:
                    self.stats['failed_requests'] += 1
                return page_num, None
                
            except Exception as e:
                async with self.stats_lock:
                    self.stats['failed_requests'] += 1
                return page_num, None
            finally:
                async with self.stats_lock:
                    self.stats['concurrent_requests'] -= 1

    async def process_chunk(self, chunk_idx: int, chunk_images: List[Path], total_chunks: int) -> Tuple[Dict[int, str], List[int]]:
        """단일 chunk를 처리하는 함수"""
        chunk_results = {}
        chunk_failed = []
        chunk_start_time = time.time()
        
        print(f"\n📦 Chunk {chunk_idx + 1}/{total_chunks} 시작 ({len(chunk_images)}개 페이지)")
        
        # Chunk 내부의 진행상황 추적
        completed_in_chunk = 0
        progress_lock = asyncio.Lock()
        
        async def process_single_task(session: aiohttp.ClientSession, image_path: Path, page_num: int):
            nonlocal completed_in_chunk
            
            try:
                result = await self.convert_single_image(session, image_path, page_num)
                page_num_result, markdown_text = result
                
                async with progress_lock:
                    completed_in_chunk += 1
                    
                    if markdown_text and markdown_text.strip():
                        chunk_results[page_num_result] = markdown_text
                        print(f"✅ [Chunk {chunk_idx + 1}] 페이지 {page_num_result} 완료 ({completed_in_chunk}/{len(chunk_images)}) - {len(markdown_text)} 문자")
                    else:
                        chunk_results[page_num_result] = f"<!-- 페이지 {page_num_result} 변환 실패: 빈 응답 -->"
                        chunk_failed.append(page_num_result)
                        print(f"❌ [Chunk {chunk_idx + 1}] 페이지 {page_num_result} 실패 ({completed_in_chunk}/{len(chunk_images)}) - 빈 응답")
                        
            except Exception as e:
                async with progress_lock:
                    completed_in_chunk += 1
                    chunk_results[page_num] = f"<!-- 페이지 {page_num} 변환 실패: {str(e)} -->"
                    chunk_failed.append(page_num)
                    print(f"❌ [Chunk {chunk_idx + 1}] 페이지 {page_num} 예외: {str(e)} ({completed_in_chunk}/{len(chunk_images)})")
        
        # Chunk 내 모든 작업을 비동기로 실행
        async with aiohttp.ClientSession() as session:
            tasks = [
                process_single_task(session, path, chunk_idx * self.chunk_size + i + 1)
                for i, path in enumerate(chunk_images)
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        chunk_time = time.time() - chunk_start_time
        success_count = len(chunk_results) - len(chunk_failed)
        print(f"📦 Chunk {chunk_idx + 1} 완료: {success_count}/{len(chunk_images)} 성공, {chunk_time:.1f}초")
        
        return chunk_results, chunk_failed

    async def convert_images_to_markdown_parallel(self, image_paths: List[Path]) -> str:
        """
        여러 이미지를 chunk 단위 배치로 비동기 마크다운 변환 (메모리 최적화)
        """
        self.reset_stats()
        total_pages = len(image_paths)
        total_chunks = (total_pages + self.chunk_size - 1) // self.chunk_size
        
        print(f"🚀 Chunk 배치 처리 시작: {total_pages}개 페이지")
        print(f"📦 Chunk 설정: {self.chunk_size}개씩 {total_chunks}개 배치, 배치당 최대 {self.max_concurrent}개 동시 처리")
        
        results = {}
        failed_pages = []
        start_time = time.time()
        overall_completed = 0
        
        # 이미지를 chunk 단위로 분할
        chunks = []
        for i in range(0, total_pages, self.chunk_size):
            chunk = image_paths[i:i + self.chunk_size]
            chunks.append(chunk)
            
        print(f"📋 Chunk 분할 완료: {len(chunks)}개 배치")
        
        # 각 chunk를 순차적으로 처리 (메모리 관리를 위해)
        for chunk_idx, chunk_images in enumerate(chunks):
            chunk_results, chunk_failed = await self.process_chunk(chunk_idx, chunk_images, total_chunks)
            
            # 결과를 전체 결과에 병합
            results.update(chunk_results)
            failed_pages.extend(chunk_failed)
            overall_completed += len(chunk_images)
            
            # 전체 진행상황 출력
            elapsed_time = time.time() - start_time
            progress_percent = (overall_completed / total_pages) * 100
            remaining_chunks = total_chunks - (chunk_idx + 1)
            estimated_remaining_time = (elapsed_time / (chunk_idx + 1)) * remaining_chunks if chunk_idx > 0 else 0
            
            print(f"\n🎯 전체 진행상황: {overall_completed}/{total_pages} ({progress_percent:.1f}%)")
            print(f"⏱️ 경과시간: {elapsed_time/60:.1f}분, 남은 Chunk: {remaining_chunks}개")
            if remaining_chunks > 0:
                print(f"📈 예상 남은 시간: {estimated_remaining_time/60:.1f}분")
            
            # Chunk 간 짧은 휴식 (메모리 정리 시간)
            if chunk_idx < total_chunks - 1:
                print(f"⏸️ 다음 Chunk 준비 중... (1초 대기)")
                await asyncio.sleep(1)

        print(f"\n🎉 모든 Chunk 처리 완료!")
        
        # 결과를 페이지 순서대로 정렬하여 마크다운 생성
        markdown_content = []
        for page_num in sorted(results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(results[page_num])
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Chunk 배치 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"  📦 처리된 Chunk: {total_chunks}개 배치")
        print(f"  📈 처리량: {len(image_paths) / total_time:.2f} 페이지/초")
        print(f"  ✅ 성공: {self.stats['successful_requests']}/{self.stats['total_requests']}")
        print(f"  ❌ 실패: {self.stats['failed_requests']}/{self.stats['total_requests']}")
        if self.stats['successful_requests'] > 0:
            avg_time = self.stats['total_time'] / self.stats['successful_requests']
            print(f"  ⚡ 평균 응답 시간: {avg_time:.2f}초")
            print(f"  📦 Chunk당 평균 시간: {total_time/total_chunks:.1f}초")
        if failed_pages:
            print(f"  ⚠️ 실패한 페이지: {sorted(failed_pages)}")
        
        # 최종 통계
        print(f"\n🏗️ Xinference 서버 성능 요약:")
        if self.stats['successful_requests'] > 0:
            avg_response_time = self.stats['total_time'] / self.stats['successful_requests']
            print(f"  {self.base_url}: {self.stats['successful_requests']}개 성공, 평균 {avg_response_time:.1f}초")
            print(f"  Chunk 기반 배치 처리로 메모리 효율성 최적화 완료")
        
        return "\n".join(markdown_content)
    
    def post_process_syncfusion_content(self, markdown_content: str, pdf_name: str) -> str:
        """Syncfusion SDK 매뉴얼 콘텐츠 후처리"""
        if not config.SYNCFUSION_MODE:
            return markdown_content
            
        processed_content = []
        
        if config.INCLUDE_METADATA:
            metadata = f"""
---
title: "{pdf_name} - Syncfusion SDK Documentation"
type: "api-documentation"
framework: "syncfusion"
version: "v11"
extracted_date: "{time.time()}"
optimized_for: ["llm-training", "rag-retrieval"]
chunk_batch_processed: true
processing_stats:
  total_requests: {self.stats['total_requests']}
  successful_requests: {self.stats['successful_requests']}
  failed_requests: {self.stats['failed_requests']}
  chunk_size: {self.chunk_size}
---

"""
            processed_content.append(metadata)
        
        processed_content.append(markdown_content)
        return '\n'.join(processed_content)

# 호환성을 위한 별칭
AsyncXinferenceClient = ChunkedAsyncXinferenceClient
ParallelOllamaClient = ChunkedAsyncXinferenceClient
AsyncParallelOllamaClient = ChunkedAsyncXinferenceClient

async def main():
    """비동기 테스트 함수"""
    client = ChunkedAsyncXinferenceClient()
    
    if await client.check_xinference_connection():
        print("✅ Xinference 서버 연결 성공")
        
        if await client.check_model_availability():
            print(f"✅ 모델 '{client.model_name}' 사용 가능 (UID: {client.model_uid})")
            print(f"🔧 Chunk 배치 설정: 크기 {client.chunk_size}, 최대 {client.max_concurrent}개 동시 요청")
        else:
            print(f"❌ 모델 '{client.model_name}' 사용 불가능")
    else:
        print("❌ Xinference 서버 연결 실패")

if __name__ == "__main__":
    asyncio.run(main())