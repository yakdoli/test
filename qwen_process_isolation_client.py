"""
Qwen2.5-VL 프로세스 격리 병렬 처리 클라이언트 (Xinference + qwen-vl-utils)

- 각 워커는 독립 프로세스로 격리되어 Xinference OpenAI 호환 API를 호출
- qwen-vl-utils를 활용해 입력 메시지/이미지 유효성 확인(로컬 전처리)
- 로컬 HF 모델 로딩 제거 → 경량/안정성 향상, 서버(Xinference)에서 모델 관리
"""

import asyncio
import multiprocessing
import os
import pickle
import psutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from modules.utils.prompt_utils import build_syncfusion_prompt
from datetime import datetime

import config
from modules.utils.md_staging import save_page_markdown



def process_isolated_image_batch(args_tuple) -> List[str]:
    """프로세스 격리된 이미지 배치 처리 (Xinference API 호출)"""
    from qwen_vl_utils import process_vision_info  # 유효성 확인 용도
    import base64
    import requests

    def encode_image_to_base64(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    try:
        # 인수 튜플 분해
        worker_id, batch_data = args_tuple

        # 배치 데이터 역직렬화
        image_paths: List[Path] = pickle.loads(batch_data)
        process_id = os.getpid()

        # Xinference 엔드포인트/모델 정보 (사전 해석된 모델 ID 우선)
        base_url = os.environ.get("XINFERENCE_BASE_URL", getattr(config, "XINFERENCE_BASE_URL", "http://localhost:9997"))
        model = os.environ.get("XINFERENCE_RESOLVED_MODEL") or \
            getattr(config, "XINFERENCE_MODEL_UID", None) or \
            getattr(config, "XINFERENCE_MODEL_NAME", "qwen2.5-vl-instruct")

        print(f"🔧 프로세스 {process_id} - 워커 {worker_id} 초기화 (Xinference API: {base_url})")

        results: List[str] = []
        process_start_time = time.time()

        session = requests.Session()
        timeout = getattr(config, "REQUEST_TIMEOUT", 300)
        max_retries = 3

        for idx, image_path in enumerate(image_paths):
            try:
                # qwen-vl-utils로 메시지 스키마 검증(이미지 경로 유효성 등)
                try:
                    messages_for_validation = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": build_syncfusion_prompt(image_path)}
                        ]
                    }]
                    # 반환값은 사용하지 않지만, 오류 발생 시 조기 감지 목적
                    _ = process_vision_info(messages_for_validation)
                except Exception as ve:
                    print(f"   ⚠️ 입력 유효성 경고(워커 {worker_id}): {image_path.name} - {ve}")

                # 이미지 base64 인코딩
                image_b64 = encode_image_to_base64(image_path)

                # OpenAI 호환 Chat Completions 페이로드 구성
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": build_syncfusion_prompt(image_path)},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 4000,
                    "stream": False
                }

                last_error: Optional[str] = None
                for attempt in range(1, max_retries + 1):
                    try:
                        resp = session.post(
                            f"{base_url}/v1/chat/completions",
                            json=payload,
                            timeout=timeout,
                            headers={"Content-Type": "application/json"}
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            content = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            if content and content.strip():
                                results.append(content)
                                print(
                                    f"   ✅ 프로세스 {process_id} - 워커 {worker_id}: "
                                    f"이미지 {idx+1}/{len(image_paths)} 완료 ({len(content)}자)"
                                )
                                # 페이지 단위 MD 스테이징 저장
                                try:
                                    save_page_markdown(
                                        image_path,
                                        content,
                                        mode="process_isolated",
                                        prompt=build_syncfusion_prompt(image_path),
                                        extra_meta={
                                            "client": "ProcessIsolatedQwenClient",
                                            "worker_id": worker_id,
                                            "process_id": process_id,
                                        },
                                    )
                                except Exception as _e:
                                    print(f"   ⚠️ MD 스테이징 저장 경고: {str(_e)}")
                                break
                            else:
                                last_error = "Empty response"
                                print(
                                    f"   ⚠️ 빈 응답(시도 {attempt}/{max_retries}) - {image_path.name}"
                                )
                        else:
                            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                            print(
                                f"   ❌ API 오류(시도 {attempt}/{max_retries}) - {last_error}"
                            )
                    except requests.exceptions.Timeout:
                        last_error = f"Timeout after {timeout}s"
                        print(
                            f"   ⏰ 타임아웃(시도 {attempt}/{max_retries}) - {image_path.name}: {last_error}"
                        )
                    except requests.RequestException as re:
                        last_error = f"Network error: {re}"
                        print(
                            f"   🔌 네트워크 오류(시도 {attempt}/{max_retries}) - {image_path.name}: {last_error}"
                        )

                    if attempt < max_retries:
                        delay = getattr(config, "RETRY_DELAY", 2) * attempt
                        time.sleep(delay)
                else:
                    # 모든 재시도 실패
                    fail_msg = f"<!-- 프로세스 {process_id} - 워커 {worker_id} 처리 실패: {image_path.name} - {last_error} -->"
                    results.append(fail_msg)
                    print(f"   ❌ 최종 실패 - {image_path.name}: {last_error}")

            except Exception as e:
                error_msg = f"<!-- 프로세스 {process_id} - 워커 {worker_id} 처리 실패: {image_path.name} - {str(e)} -->"
                results.append(error_msg)
                print(f"   ❌ 예외 - 이미지 {idx+1}: {e}")

        process_time = time.time() - process_start_time
        throughput = len(image_paths) / process_time if process_time > 0 else 0

        print(
            f"🎯 프로세스 {process_id} - 워커 {worker_id} 완료: "
            f"{len(results)}개, {process_time:.1f}초, {throughput:.2f} 이미지/초"
        )

        return results

    except Exception as e:
        print(f"❌ 프로세스 배치 처리 오류(워커 {locals().get('worker_id', '?')}): {e}")
        return [f"<!-- 프로세스 격리 실패: {str(e)} -->"]


# NOTE: Deprecated local prompt function (replaced by modules.utils.prompt_utils.build_syncfusion_prompt)
# def get_syncfusion_prompt(image_path: Path) -> str:
    """Syncfusion 특화 프롬프트 (프로세스 격리 최적화)
    - 미세조정 데이터셋/RAG 일관성 강화를 위해 동적 컨텍스트를 포함한 지시문
    - 컨텍스트/메타 정보 표준화, 섹션 스키마 고정, 언어/코드/표/리스트/링크 처리 규약 강화
    - OCR 정규화 규칙 포함 (띄어쓰기, 하이픈 줄바꿈, 특수문자 통합 등)
    """
    import re
    from datetime import datetime

    # 컨텍스트 추출
    file_name = image_path.name
    parent_dir = image_path.parent.name
    stem = image_path.stem
    m = re.search(r"page[_-]?(\d+)", stem, re.IGNORECASE)
    page_number = m.group(1) if m else "unknown"

    # ISO-8601 UTC 타임스탬프 (초 단위)
    iso_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # 모델에 전달할 컨텍스트 + 엄격한 출력 계약
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


class ProcessIsolationResourceManager:
    """프로세스 격리 리소스 관리자"""
    
    def __init__(self):
        import torch
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        self.cpu_count = multiprocessing.cpu_count()
        self.system_memory = psutil.virtual_memory()
        
    def get_optimal_process_config(self) -> Dict[str, Any]:
        """최적 프로세스 격리 설정
        - Xinference 사용 시 클라이언트 프로세스는 GPU 불필요 → CPU 기반 병렬화
        - 로컬 모델 사용 전환 시에는 GPU 개수 기준으로 제한
        """
        # 기본 권장 프로세스 수 계산
        cpu_based = max(1, min(self.cpu_count // 2, getattr(config, 'MAX_WORKERS', 8)))
        gpu_based = max(1, self.device_count) if self.gpu_available else 1

        # 모드에 따른 결정
        if getattr(config, 'USE_DIRECT_QWEN', False):
            recommended_processes = min(gpu_based, cpu_based)
        else:
            # Xinference 모드: CPU 기반 동시성 사용 (I/O 바운드 HTTP)
            recommended_processes = cpu_based

        print(f"🔧 프로세스 격리 리소스 분석:")
        print(f"   GPU 개수: {self.device_count}")
        print(f"   CPU 코어: {self.cpu_count}")
        print(f"   시스템 RAM: {self.system_memory.total / (1024**3):.1f}GB")
        print(f"   권장 프로세스 수: {recommended_processes}")
        
        return {
            'max_processes': recommended_processes,
            'gpus_available': self.device_count,
            'isolation_mode': 'complete_process_isolation',
            'memory_isolation': True
        }


class ProcessIsolatedQwenClient:
    """프로세스 격리 Qwen2.5-VL 클라이언트"""
    
    def __init__(self):
        # spawn 방식으로 멀티프로세싱 설정 (CUDA 호환)
        multiprocessing.set_start_method('spawn', force=True)
        
        self.resource_manager = ProcessIsolationResourceManager()
        self.process_pool = None
        self.workers_initialized = False
        self.xinference_model_id: Optional[str] = None  
        self.stats = {
            'total_images': 0,
            'total_processing_time': 0,
            'process_stats': {},
            'isolation_overhead': 0
        }
        
    async def initialize_isolated_system(self) -> bool:
        """프로세스 격리 시스템 초기화"""
        print("🚀 프로세스 격리 시스템 초기화 중...")
        
        try:
            config = self.resource_manager.get_optimal_process_config()
            max_processes = config['max_processes']
            
            # ProcessPoolExecutor 생성 (spawn context)
            ctx = multiprocessing.get_context('spawn')
            self.process_pool = ProcessPoolExecutor(
                max_workers=max_processes,
                mp_context=ctx
            )
            
            print(f"✅ 프로세스 격리 시스템 초기화 완료: {max_processes}개 독립 프로세스")
            self.workers_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 프로세스 격리 시스템 초기화 실패: {e}")
            return False
    
    async def convert_images_process_isolated(self, image_paths: List[Path]) -> str:
        """프로세스 격리 이미지 변환 (Xinference API 호출 기반)"""
        if not self.workers_initialized:
            if not await self.initialize_isolated_system():
                return "프로세스 격리 시스템 초기화 실패"

        # 사전 헬스체크 및 모델 ID 해석
        try:
            import requests
            base_url = getattr(config, "XINFERENCE_BASE_URL", "http://localhost:9997")
            resp = requests.get(f"{base_url}/v1/models", timeout=10)
            if resp.status_code != 200:
                return f"<!-- Xinference 헬스체크 실패: HTTP {resp.status_code} -->"
            data = resp.json()
            models = data.get('data', []) if isinstance(data, dict) else []
            name = getattr(config, 'XINFERENCE_MODEL_NAME', 'qwen2.5-vl-instruct')
            found = None
            for m in models:
                mid = m.get('id') or m.get('model')
                if not mid:
                    continue
                if mid.startswith(name):
                    found = mid
                    break
            if not found and getattr(config, 'XINFERENCE_MODEL_UID', None):
                found = getattr(config, 'XINFERENCE_MODEL_UID')
            if not found:
                print("❌ Xinference에 대상 모델이 로드되어 있지 않습니다. start_xinference.sh 실행 후 /v1/models에 모델이 보여야 합니다.")
                return "<!-- Xinference 모델 미로딩: start_xinference.sh로 모델 구동 후 재시도 -->"
            self.xinference_model_id = found
            # 워커에 환경변수로 전달(스폰 시 상속)
            os.environ['XINFERENCE_BASE_URL'] = base_url
            os.environ['XINFERENCE_RESOLVED_MODEL'] = self.xinference_model_id
            print(f"🔗 Xinference 모델 확인: {self.xinference_model_id}")
        except Exception as e:
            print(f"⚠️ Xinference 모델 확인 중 예외: {e}")
            # 계속 시도하되, 서버 측에서 500 발생 시 재시도 로직이 처리
        
        total_images = len(image_paths)
        # 워커 수 기준 균등 분할 (Xinference 모드에서는 GPU와 무관)
        max_workers = self.process_pool._max_workers if self.process_pool else 1
        safe_concurrency = getattr(config, 'MAX_CONCURRENT_REQUESTS', 6)
        num_batches = min(max_workers, total_images, safe_concurrency) if total_images > 0 else 0

        batches: List[Tuple[int, List[Path]]] = []
        for i in range(num_batches):
            batches.append((i, []))
        for idx, p in enumerate(image_paths):
            batches[idx % num_batches][1].append(p)

        print(f"🚀 프로세스 격리 변환 시작:")
        print(f"   총 이미지: {total_images}개")
        print(f"   독립 프로세스: {len(batches)}개")
        print(f"   모드: {'Xinference API' if not getattr(config, 'USE_DIRECT_QWEN', False) else 'Direct HF Model'}")
        if self.xinference_model_id:
            print(f"   대상 모델 ID: {self.xinference_model_id}")
        
        start_time = datetime.now()
        
        # 각 배치를 독립 프로세스에서 처리
        loop = asyncio.get_event_loop()
        tasks = []
        
        for worker_id, batch in batches:
            # 배치 데이터 직렬화
            batch_data = pickle.dumps(batch)
            
            # 독립 프로세스 실행 (GPU ID와 배치 데이터를 튜플로 전달)
            task = loop.run_in_executor(
                self.process_pool,
                process_isolated_image_batch,
                (worker_id, batch_data)
            )
            tasks.append((task, worker_id, len(batch)))
        
        print(f"⚡ {len(tasks)}개 완전 격리 프로세스 실행 중...")
        
        # 모든 프로세스 완료 대기
        all_results = {}
        page_counter = 1
        
        for task, worker_id, batch_size in tasks:
            try:
                batch_results = await task
                
                for result in batch_results:
                    all_results[page_counter] = result
                    page_counter += 1
                
                print(f"✅ 프로세스 격리 완료 - 워커 {worker_id}: {batch_size}개 처리")
                
            except Exception as e:
                print(f"❌ 프로세스 격리 실패 - 워커 {worker_id}: {e}")
                for _ in range(batch_size):
                    all_results[page_counter] = f"<!-- 프로세스 격리 실패 - 워커 {worker_id} -->"
                    page_counter += 1
        
        # 결과 조합
        markdown_content = []
        for page_num in sorted(all_results.keys()):
            if page_num > 1:
                markdown_content.append("\n---\n")
            markdown_content.append(f"<!-- 페이지 {page_num} -->\n")
            markdown_content.append(all_results[page_num])
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 성능 통계 출력
        print(f"\n📊 프로세스 격리 처리 완료:")
        print(f"  ⏱️ 총 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"  🚀 처리량: {total_images / total_time:.2f} 이미지/초")
        print(f"  🔒 격리 수준: 완전 프로세스 격리")
        print(f"  📈 병렬 효율성: {len(batches)}개 독립 프로세스")
        print(f"  🎯 메모리 격리: 프로세스별 완전 분리")
        
        self.stats.update({
            'total_images': total_images,
            'total_processing_time': total_time,
            'processes_used': len(batches),
            'throughput': total_images / total_time,
            'isolation_level': 'complete'
        })
        
        return "\n".join(markdown_content)
    
    def get_isolation_stats(self) -> Dict[str, Any]:
        """프로세스 격리 통계"""
        return {
            'mode': 'process_isolated_parallel',
            'isolation_level': 'complete_process_separation',
            'gpu_count': self.resource_manager.device_count,
            'memory_isolation': True,
            'performance_stats': self.stats
        }
    
    def cleanup(self):
        """프로세스 격리 시스템 정리"""
        print("🧹 프로세스 격리 시스템 정리 중...")
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        self.workers_initialized = False
        print("✅ 프로세스 격리 시스템 정리 완료")


async def main():
    """테스트 함수"""
    client = ProcessIsolatedQwenClient()
    
    print("🧪 프로세스 격리 클라이언트 테스트")
    if await client.initialize_isolated_system():
        print("✅ 프로세스 격리 시스템 초기화 성공")
        
        stats = client.get_isolation_stats()
        print(f"📊 시스템 설정: {stats['mode']}")
        print(f"🔒 격리 수준: {stats['isolation_level']}")
        print(f"🎯 GPU 수: {stats['gpu_count']}")
        
        client.cleanup()
    else:
        print("❌ 프로세스 격리 시스템 초기화 실패")


if __name__ == "__main__":
    asyncio.run(main())