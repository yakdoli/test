"""
통합 Ollama/Qwen 클라이언트
Xinference와 직접 Qwen2.5-VL 사용을 설정에 따라 선택
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import config

# 조건부 임포트
if config.USE_DIRECT_QWEN:
    from qwen_direct_client import DirectQwenVLClient
else:
    from parallel_ollama_client import ChunkedAsyncXinferenceClient


class UnifiedVLClient:
    """통합 비전-언어 모델 클라이언트"""
    
    def __init__(self):
        self.use_direct_qwen = config.USE_DIRECT_QWEN
        
        if self.use_direct_qwen:
            print("🎯 Direct Qwen2.5-VL-7B-Instruct 모드 활성화")
            self.client = DirectQwenVLClient()
        else:
            print("🌐 Xinference API 모드 활성화")
            self.client = ChunkedAsyncXinferenceClient()
        
        self.stats = {
            'mode': 'direct_qwen' if self.use_direct_qwen else 'xinference',
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0
        }
    
    async def initialize(self) -> bool:
        """클라이언트 초기화"""
        if self.use_direct_qwen:
            return await self.client.initialize_model()
        else:
            # Xinference 클라이언트는 별도 초기화가 필요없음
            return await self.client.check_xinference_connection()
    
    async def check_availability(self) -> bool:
        """모델 사용 가능성 확인"""
        if self.use_direct_qwen:
            return self.client.model is not None
        else:
            return await self.client.check_model_availability()
    
    async def convert_images_to_markdown_parallel(self, image_paths: List[Path]) -> str:
        """이미지들을 병렬로 마크다운 변환"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await self.client.convert_images_to_markdown_parallel(image_paths)
            
            # 통계 업데이트
            self.stats['total_requests'] += len(image_paths)
            self.stats['total_time'] += asyncio.get_event_loop().time() - start_time
            
            if result and result.strip():
                self.stats['successful_requests'] += len(image_paths)
            else:
                self.stats['failed_requests'] += len(image_paths)
            
            return result
            
        except Exception as e:
            self.stats['failed_requests'] += len(image_paths)
            print(f"❌ 이미지 변환 실패: {e}")
            return f"<!-- 변환 실패: {str(e)} -->"
    
    def post_process_syncfusion_content(self, markdown_content: str, pdf_name: str) -> str:
        """Syncfusion 콘텐츠 후처리 (기존 메서드 호환성)"""
        if hasattr(self.client, 'post_process_syncfusion_content'):
            return self.client.post_process_syncfusion_content(markdown_content, pdf_name)
        else:
            # 직접 후처리 구현
            if not config.SYNCFUSION_MODE:
                return markdown_content
                
            processed_content = []
            
            if config.INCLUDE_METADATA:
                metadata = f"""---
title: "{pdf_name} - Syncfusion SDK Documentation"
type: "api-documentation"
framework: "syncfusion"
version: "v11"
extracted_date: "{asyncio.get_event_loop().time()}"
optimized_for: ["llm-training", "rag-retrieval"]
processing_mode: "{self.stats['mode']}"
processing_stats:
  total_requests: {self.stats['total_requests']}
  successful_requests: {self.stats['successful_requests']}
  failed_requests: {self.stats['failed_requests']}
---

"""
                processed_content.append(metadata)
            
            processed_content.append(markdown_content)
            return '\n'.join(processed_content)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.stats.copy()
        
        if self.use_direct_qwen and hasattr(self.client, 'stats'):
            # Direct Qwen 클라이언트의 상세 통계 병합
            direct_stats = self.client.stats
            stats.update({
                'detailed_stats': direct_stats,
                'average_processing_time': (
                    direct_stats['total_processing_time'] / direct_stats['successful_requests']
                    if direct_stats['successful_requests'] > 0 else 0
                )
            })
        elif not self.use_direct_qwen and hasattr(self.client, 'stats'):
            # Xinference 클라이언트의 상세 통계 병합
            xinference_stats = self.client.stats
            stats.update({
                'detailed_stats': xinference_stats,
                'chunk_size': self.client.chunk_size,
                'max_concurrent': self.client.max_concurrent
            })
        
        return stats
    
    def cleanup(self):
        """리소스 정리"""
        if self.use_direct_qwen and hasattr(self.client, 'cleanup'):
            self.client.cleanup()
        
        print(f"🧹 {self.stats['mode']} 클라이언트 정리 완료")


# 호환성을 위한 별칭들
AsyncXinferenceClient = UnifiedVLClient
ParallelOllamaClient = UnifiedVLClient
AsyncParallelOllamaClient = UnifiedVLClient


async def main():
    """테스트 함수"""
    print(f"🧪 통합 클라이언트 테스트 시작 (모드: {'Direct Qwen' if config.USE_DIRECT_QWEN else 'Xinference'})")
    
    client = UnifiedVLClient()
    
    if await client.initialize():
        print("✅ 클라이언트 초기화 성공")
        
        if await client.check_availability():
            print("✅ 모델 사용 가능")
            
            # 성능 통계 출력
            stats = client.get_performance_stats()
            print(f"📊 모드: {stats['mode']}")
            
        else:
            print("❌ 모델 사용 불가")
    else:
        print("❌ 클라이언트 초기화 실패")
    
    client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())