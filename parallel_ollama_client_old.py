"""
Xinference API를 사용하는 클라이언트 (단일 인스턴스)
"""

import asyncio
import aiohttp
import json
import base64
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import config

class AsyncXinferenceClient:
    def __init__(self):
        self.base_url = config.XINFERENCE_BASE_URL
        self.model_name = config.XINFERENCE_MODEL_NAME
        self.model_uid = config.XINFERENCE_MODEL_UID
        self.max_concurrent = config.MAX_CONCURRENT_REQUESTS
        self.request_timeout = config.REQUEST_TIMEOUT
        self.retry_delay = config.RETRY_DELAY
        
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
                        print(f"🔄 페이지 {page_num} 처리 시작 (시도: {attempt + 1}/{max_retries})")
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
                                    print(f"✅ 페이지 {page_num} 성공 ({len(markdown_text)} 문자)")
                                    return page_num, markdown_text
                                else:
                                    print(f"⚠️ 페이지 {page_num} 빈 응답 수신")
                                    last_exception = "Empty response received"
                            else:
                                response_text = await response.text()
                                error_msg = f"HTTP {response.status}: {response_text[:200]}"
                                print(f"❌ API 오류 (페이지: {page_num}, 시도: {attempt + 1}): {error_msg}")
                                last_exception = error_msg
                                
                    except asyncio.TimeoutError as e:
                        error_msg = f"Timeout after {self.request_timeout}s"
                        print(f"⏰ 타임아웃 (페이지: {page_num}, 시도: {attempt + 1}): {error_msg}")
                        last_exception = error_msg
                    except aiohttp.ClientError as e:
                        error_msg = f"Network error: {str(e)}"
                        print(f"🔌 네트워크 오류 (페이지: {page_num}, 시도: {attempt + 1}): {error_msg}")
                        last_exception = error_msg
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        print(f"❌ 예상치 못한 오류 (페이지: {page_num}, 시도: {attempt + 1}): {error_msg}")
                        last_exception = error_msg
                    
                    # 재시도 대기 (마지막 시도가 아닌 경우)
                    if attempt < max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        print(f"⏳ {wait_time}초 대기 후 재시도...")
                        await asyncio.sleep(wait_time)
                
                # 모든 재시도 실패
                print(f"❌ 페이지 {page_num} 최종 실패: {last_exception}")
                async with self.stats_lock:
                    self.stats['failed_requests'] += 1
                return page_num, None
                
            except Exception as e:
                print(f"❌ 심각한 오류 (페이지 {page_num}): {e}")
                async with self.stats_lock:
                    self.stats['failed_requests'] += 1
                return page_num, None
            finally:
                async with self.stats_lock:
                    self.stats['concurrent_requests'] -= 1
    
    async def convert_images_to_markdown_parallel(self, image_paths: List[Path]) -> str:
        """
        여러 이미지를 비동기 병렬로 마크다운으로 변환 (고급 진행상황 모니터링)
        """
        self.reset_stats()
        print(f"🚀 비동기 처리 시작: {len(image_paths)}개 페이지, 최대 {self.max_concurrent}개 동시 처리")
        
        results = {}
        failed_pages = []
        start_time = time.time()
        
        # 진행상황 추적 변수
        completed_count = 0
        total_pages = len(image_paths)
        last_progress_time = start_time
        
        # 진행상황 모니터링을 위한 락
        progress_lock = asyncio.Lock()
        
        async def print_progress_stats():
            """주기적으로 진행상황과 워커 통계를 출력"""
            while completed_count < total_pages:
                await asyncio.sleep(10)  # 10초마다 통계 출력
                
                async with progress_lock:
                    if completed_count < total_pages:
                        elapsed_time = time.time() - start_time
                        progress_percent = (completed_count / total_pages) * 100
                        pages_per_sec = completed_count / elapsed_time if elapsed_time > 0 else 0
                        estimated_total_time = elapsed_time / completed_count * total_pages if completed_count > 0 else 0
                        remaining_time = estimated_total_time - elapsed_time
                        
                        print(f"\n📊 진행상황: {completed_count}/{total_pages} ({progress_percent:.1f}%)")
                        print(f"⏱️ 경과시간: {elapsed_time/60:.1f}분, 예상 남은 시간: {remaining_time/60:.1f}분")
                        print(f"📈 처리속도: {pages_per_sec:.1f} 페이지/초")
                        
                        print(f"🏗️ Xinference 서버: 정상 작동 중")
        
        async def process_single_task(session: aiohttp.ClientSession, image_path: Path, page_num: int):
            """단일 작업 처리 및 진행상황 업데이트"""
            nonlocal completed_count, last_progress_time
            
            try:
                print(f"🎯 페이지 {page_num} 작업 시작 ({image_path.name})")
                result = await self.convert_single_image(session, image_path, page_num)
                page_num_result, markdown_text = result
                
                async with progress_lock:
                    completed_count += 1
                    current_time = time.time()
                    
                    if markdown_text and markdown_text.strip():
                        results[page_num_result] = markdown_text
                        elapsed_time = current_time - start_time
                        progress_percent = (completed_count / total_pages) * 100
                        estimated_total_time = elapsed_time / completed_count * total_pages if completed_count > 0 else 0
                        remaining_time = max(0, estimated_total_time - elapsed_time)
                        
                        print(f"✅ 페이지 {page_num_result} 완료 ({completed_count}/{total_pages}, {progress_percent:.1f}%) "
                              f"- 예상 남은 시간: {remaining_time/60:.1f}분")
                        last_progress_time = current_time
                    else:
                        results[page_num_result] = f"<!-- 페이지 {page_num_result} 변환 실패: 빈 응답 -->"
                        failed_pages.append(page_num_result)
                        print(f"❌ 페이지 {page_num_result} 실패 ({completed_count}/{total_pages}) - 빈 응답")
                    
            except Exception as e:
                async with progress_lock:
                    completed_count += 1
                    results[page_num] = f"<!-- 페이지 {page_num} 변환 실패: {str(e)} -->"
                    failed_pages.append(page_num)
                    print(f"❌ 페이지 {page_num} 예외 발생: {str(e)} ({completed_count}/{total_pages})")
                    
            print(f"🏁 페이지 {page_num} 작업 완료")
        
        # aiohttp 세션을 사용하여 모든 작업을 동시에 실행
        async with aiohttp.ClientSession() as session:
            # 모든 작업을 비동기 태스크로 생성
            tasks = [
                process_single_task(session, path, i + 1)
                for i, path in enumerate(image_paths)
            ]
            
            print(f"📋 {len(tasks)}개 작업 생성 완료, 실행 중...")
            
            # 진행상황 모니터링 태스크 시작
            progress_monitor = asyncio.create_task(print_progress_stats())
            
            try:
                # 작업을 동시에 실행하되, 모든 작업이 완료될 때까지 기다림
                print(f"⏳ 모든 작업 실행 대기 중...")
                results_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 예외가 발생한 작업들 확인
                for i, result in enumerate(results_list):
                    if isinstance(result, Exception):
                        page_num = i + 1
                        print(f"❗ 작업 {page_num}에서 예외 발생: {result}")
                        async with progress_lock:
                            if page_num not in results:
                                results[page_num] = f"<!-- 페이지 {page_num} 변환 실패: {str(result)} -->"
                                failed_pages.append(page_num)
                                
                print(f"🎉 모든 작업 완료: 성공 {len(results) - len(failed_pages)}/{len(results)}")
                        
            finally:
                # 진행상황 모니터링 태스크 종료
                progress_monitor.cancel()
                try:
                    await progress_monitor
                except asyncio.CancelledError:
                    pass
        
        # 결과를 페이지 순서대로 정렬하여 마크다운 생성
        markdown_content = []
        for page_num in sorted(results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(results[page_num])
        
        total_time = time.time() - start_time
        
        print(f"\n📊 비동기 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"  📈 처리량: {len(image_paths) / total_time:.2f} 페이지/초")
        print(f"  ✅ 성공: {self.stats['successful_requests']}/{self.stats['total_requests']}")
        print(f"  ❌ 실패: {self.stats['failed_requests']}/{self.stats['total_requests']}")
        if self.stats['successful_requests'] > 0:
            avg_time = self.stats['total_time'] / self.stats['successful_requests']
            print(f"  ⚡ 평균 응답 시간: {avg_time:.2f}초")
        if failed_pages:
            print(f"  ⚠️ 실패한 페이지: {sorted(failed_pages)}")
        
        # 최종 통계
        print(f"\n🏗️ Xinference 서버 성능 요약:")
        if self.stats['successful_requests'] > 0:
            avg_response_time = self.stats['total_time'] / self.stats['successful_requests']
            print(f"  {self.base_url}: {self.stats['successful_requests']}개 성공, 평균 {avg_response_time:.1f}초")
        
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
async_parallel_processed: true
processing_stats:
  total_requests: {self.stats['total_requests']}
  successful_requests: {self.stats['successful_requests']}
  failed_requests: {self.stats['failed_requests']}
---

"""
            processed_content.append(metadata)
        
        processed_content.append(markdown_content)
        return '\n'.join(processed_content)

# 호환성을 위한 별칭
ParallelOllamaClient = AsyncXinferenceClient
AsyncParallelOllamaClient = AsyncXinferenceClient

async def main():
    """비동기 테스트 함수"""
    client = AsyncXinferenceClient()
    
    if await client.check_xinference_connection():
        print("✅ Xinference 서버 연결 성공")
        
        if await client.check_model_availability():
            print(f"✅ 모델 '{client.model_name}' 사용 가능 (UID: {client.model_uid})")
            print(f"🔧 비동기 설정: 최대 {client.max_concurrent}개 동시 요청")
        else:
            print(f"❌ 모델 '{client.model_name}' 사용 불가능")
    else:
        print("❌ Xinference 서버 연결 실패")

if __name__ == "__main__":
    asyncio.run(main())
