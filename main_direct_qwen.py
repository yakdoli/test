"""
Qwen2.5-VL 직접 로드 통합 메인 프로그램
세 가지 스케일링 접근 방식을 지원:
1. Scale-up: 전용 GPU 로드 (단일 GPU 최적화)
2. Scale-out: 다중 GPU 워커 분배 (여러 GPU 병렬)
3. Process isolation: 프로세스 격리 병렬 처리 (완전 격리)
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Optional

import config
from pdf_converter import PDFConverter


class DirectQwenPDFConverter:
    """Qwen2.5-VL 직접 로드 PDF 변환기 통합 클래스"""
    
    def __init__(self):
        self.pdf_converter = PDFConverter()
        self.output_dir = config.OUTPUT_DIR
        self.client = None
        self.scaling_mode = None
        
    def select_scaling_approach(self) -> str:
        """스케일링 접근 방식 선택"""
        import torch
        
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        print("🎯 Qwen2.5-VL 직접 로드 스케일링 방식 선택:")
        print("=" * 60)
        
        if gpu_count == 0:
            print("❌ GPU가 감지되지 않았습니다. CPU 모드는 지원되지 않습니다.")
            sys.exit(1)
        
        print(f"🔧 시스템 정보: {gpu_count}개 GPU 감지됨")
        print()
        
        print("사용 가능한 스케일링 방식:")
        print("1️⃣ Scale-up (전용 GPU 로드)")
        print("   - 단일 GPU에서 모델 완전 로드")
        print("   - 최고 성능, 메모리 효율성")
        print("   - 권장: 고성능 단일 GPU (24GB+ VRAM)")
        print()
        
        print("2️⃣ Scale-out (다중 GPU 분산)")
        print("   - 여러 GPU에 모델 분산 로드")
        print("   - 메모리 분산, 높은 처리량")
        print("   - 권장: 다중 GPU 시스템 (2개+ GPU)")
        print()
        
        print("3️⃣ Process Isolation (프로세스 격리)")
        print("   - GPU별 독립 프로세스 실행")
        print("   - 완전 메모리 격리, 안정성")
        print("   - 권장: 다중 GPU + 높은 안정성 필요")
        print()
        
        # 자동 권장 방식
        if gpu_count == 1:
            recommended = "1"
            print("🎯 권장 방식: Scale-up (단일 GPU 최적화)")
        elif gpu_count >= 4:
            recommended = "3"
            print("🎯 권장 방식: Process Isolation (다중 GPU 안정성)")
        else:
            recommended = "2"
            print("🎯 권장 방식: Scale-out (다중 GPU 분산)")
        
        # 자동 권장 방식 (항상 Process Isolation 선택)
        recommended = "3"
        print("🎯 권장 방식: Process Isolation (다중 GPU 안정성) - 자동 선택됨")
        choice = recommended
        return choice
    
    async def initialize_client(self, mode: str) -> bool:
        """선택한 모드에 따라 클라이언트 초기화"""
        self.scaling_mode = mode
        
        try:
            if mode == "1":
                # Scale-up: 전용 GPU 로드
                from qwen_direct_client import DirectQwenVLClient
                self.client = DirectQwenVLClient()
                print("🚀 Scale-up 모드: 전용 GPU 로드 클라이언트 초기화 중...")
                return await self.client.initialize_model()
                
            elif mode == "2":
                # Scale-out: 다중 GPU 분산
                from qwen_multi_gpu_client import OptimizedMultiGPUQwenClient
                self.client = OptimizedMultiGPUQwenClient()
                print("🚀 Scale-out 모드: 다중 GPU 분산 클라이언트 초기화 중...")
                return await self.client.initialize_model()
                
            elif mode == "3":
                # Process Isolation: 프로세스 격리
                from qwen_process_isolation_client import ProcessIsolatedQwenClient
                self.client = ProcessIsolatedQwenClient()
                print("🚀 Process Isolation 모드: 프로세스 격리 클라이언트 초기화 중...")
                return await self.client.initialize_isolated_system()
                
            else:
                print(f"❌ 지원하지 않는 모드: {mode}")
                return False
                
        except Exception as e:
            print(f"❌ 클라이언트 초기화 실패: {e}")
            return False
    
    async def check_prerequisites(self, specific_pdf: Optional[str] = None) -> bool:
        """실행 전 필수 조건 확인"""
        print("🔍 실행 환경 확인 중...")
        
        # PDF 디렉토리 존재 확인
        if not config.PDF_DIR.exists():
            print(f"❌ PDF 디렉토리가 존재하지 않습니다: {config.PDF_DIR}")
            return False
        
        # PDF 파일 존재 확인
        if specific_pdf:
            pdf_files = [config.PDF_DIR / f"{specific_pdf}.pdf"]
            if not pdf_files[0].exists():
                print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_files[0]}")
                return False
        else:
            pdf_files = list(config.PDF_DIR.glob("*.pdf"))
            if not pdf_files:
                print(f"❌ PDF 파일을 찾을 수 없습니다: {config.PDF_DIR}")
                return False
        
        print(f"✅ {len(pdf_files)}개의 PDF 파일 발견")
        
        # 이미지 파일 확인
        if specific_pdf:
            staging_dirs = [config.STAGING_DIR / specific_pdf]
        else:
            staging_dirs = [d for d in config.STAGING_DIR.iterdir() if d.is_dir()]
        
        if not staging_dirs:
            print("❌ 스테이징 이미지가 없습니다.")
            print("   먼저 PDF를 이미지로 변환하세요: python pdf_converter.py")
            return False
        
        total_images = 0
        for staging_dir in staging_dirs:
            images = list(staging_dir.glob("*.jpeg"))
            total_images += len(images)
        
        if total_images == 0:
            print("❌ 변환할 이미지가 없습니다.")
            print("   먼저 PDF를 이미지로 변환하세요: python pdf_converter.py")
            return False
        
        print(f"✅ {total_images}개의 이미지 파일 발견")
        return True
    
    async def convert_single_pdf(self, pdf_name: str, image_paths: list) -> bool:
        """단일 PDF 변환"""
        print(f"\n📄 '{pdf_name}' 변환 시작...")
        print(f"🖼️ 총 {len(image_paths)}개 페이지")
        print(f"🎯 사용 모드: {self.get_mode_description()}")
        
        try:
            start_time = time.time()
            
            # 선택한 모드에 따라 적절한 메소드 호출
            if self.scaling_mode == "1":
                # Scale-up
                markdown_content = await self.client.convert_images_to_markdown_parallel(image_paths)
            elif self.scaling_mode == "2":
                # Scale-out
                markdown_content = await self.client.convert_images_to_markdown_parallel_optimized(image_paths)
            elif self.scaling_mode == "3":
                # Process Isolation
                markdown_content = await self.client.convert_images_process_isolated(image_paths)
            else:
                return False
            
            if not markdown_content.strip():
                print(f"❌ '{pdf_name}' 변환 실패: 빈 내용")
                return False
            
            # Syncfusion 특화 후처리
            if config.SYNCFUSION_MODE:
                markdown_content = self.post_process_syncfusion_content(markdown_content, pdf_name)
            
            # 마크다운 파일 저장
            mode_suffix = {
                "1": "scale_up",
                "2": "scale_out", 
                "3": "process_isolated"
            }
            
            output_file = self.output_dir / f"{pdf_name}_{mode_suffix[self.scaling_mode]}.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n✅ '{pdf_name}' 변환 완료!")
            print(f"📁 출력 파일: {output_file}")
            print(f"⏱️ 처리 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
            print(f"📊 처리량: {len(image_paths)/total_time:.2f} 페이지/초")
            
            return True
            
        except Exception as e:
            print(f"❌ '{pdf_name}' 변환 실패: {str(e)}")
            return False
    
    def get_mode_description(self) -> str:
        """현재 모드 설명 반환"""
        mode_desc = {
            "1": "Scale-up (전용 GPU 로드)",
            "2": "Scale-out (다중 GPU 분산)",
            "3": "Process Isolation (프로세스 격리)"
        }
        return mode_desc.get(self.scaling_mode, "Unknown")
    
    def post_process_syncfusion_content(self, content: str, pdf_name: str) -> str:
        """Syncfusion 특화 후처리"""
        if not config.SYNCFUSION_MODE:
            return content
        
        metadata = f"""---
title: "{pdf_name} - Syncfusion SDK Documentation"
type: "api-documentation"
framework: "syncfusion"
version: "v11"
processing_mode: "{self.get_mode_description()}"
extracted_date: "{time.time()}"
optimized_for: ["llm-training", "rag-retrieval"]
scaling_approach: "{self.scaling_mode}"
---

"""
        return metadata + content
    
    async def run(self, specific_pdf: Optional[str] = None):
        """전체 변환 프로세스 실행"""
        print("🚀 Qwen2.5-VL 직접 로드 PDF 변환기")
        print("=" * 60)
        
        # 실행 환경 확인
        if not await self.check_prerequisites(specific_pdf):
            print("\n❌ 실행 환경이 준비되지 않았습니다.")
            sys.exit(1)
        
        # 스케일링 방식 선택
        mode = self.select_scaling_approach()
        
        # 클라이언트 초기화
        if not await self.initialize_client(mode):
            print("\n❌ 클라이언트 초기화에 실패했습니다.")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("📸 이미지 파일 확인 및 변환 시작")
        
        # PDF별 이미지 수집
        if specific_pdf:
            staging_dir = config.STAGING_DIR / specific_pdf
            if not staging_dir.exists():
                print(f"❌ {specific_pdf} 스테이징 디렉토리가 없습니다.")
                return
            
            image_paths = sorted(list(staging_dir.glob("*.jpeg")))
            if not image_paths:
                print(f"❌ {specific_pdf} 이미지 파일이 없습니다.")
                return
            
            pdf_images = {specific_pdf: image_paths}
            print(f"🎯 특정 PDF 처리: {specific_pdf} ({len(image_paths)}개 이미지)")
        else:
            # 모든 PDF의 이미지 확인
            pdf_images = {}
            for pdf_dir in config.STAGING_DIR.iterdir():
                if pdf_dir.is_dir():
                    image_paths = sorted(list(pdf_dir.glob("*.jpeg")))
                    if image_paths:
                        pdf_images[pdf_dir.name] = image_paths
            
            if not pdf_images:
                print("❌ 변환할 이미지가 없습니다.")
                return
        
        print(f"\n📝 변환 시작 ({len(pdf_images)}개 PDF)")
        print(f"🎯 사용 모드: {self.get_mode_description()}")
        
        # 변환 실행
        success_count = 0
        total_count = len(pdf_images)
        total_start_time = time.time()
        
        for pdf_name, image_paths in pdf_images.items():
            if await self.convert_single_pdf(pdf_name, image_paths):
                success_count += 1
        
        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("🎉 Qwen2.5-VL 직접 로드 변환 완료!")
        print(f"🎯 사용 모드: {self.get_mode_description()}")
        print(f"✅ 성공: {success_count}/{total_count}")
        print(f"⏱️ 전체 시간: {total_processing_time:.2f}초 ({total_processing_time/60:.1f}분)")
        
        if success_count < total_count:
            print(f"❌ 실패: {total_count - success_count}")
        
        print(f"📁 출력 디렉토리: {self.output_dir}")
        
        # 클라이언트 정리
        if hasattr(self.client, 'cleanup'):
            self.client.cleanup()


async def main():
    """메인 함수"""
    # 간단한 CLI 옵션 처리: --xinference-base-url (직접 Qwen 모드에서도 일부 경로에서 참조될 수 있으므로 허용)
    try:
        if any(arg.startswith('--xinference-base-url') or arg == '--base-url' or arg == '--x-base-url' for arg in sys.argv[1:]):
            for i, arg in enumerate(sys.argv[1:], start=1):
                if arg.startswith('--xinference-base-url='):
                    url = arg.split('=', 1)[1]
                    config.XINFERENCE_BASE_URL = url
                elif arg in ('--xinference-base-url', '--base-url', '--x-base-url') and i + 1 < len(sys.argv):
                    url = sys.argv[i + 1]
                    config.XINFERENCE_BASE_URL = url
            # 하위 프로세스에서도 동일하게 사용하도록 환경변수 설정
            import os as _os
            _os.environ['XINFERENCE_BASE_URL'] = config.XINFERENCE_BASE_URL
            print(f"🌐 Xinference Base URL: {config.XINFERENCE_BASE_URL}")
    except Exception as e:
        print(f"⚠️ Xinference Base URL 파싱 실패: {e}")

    converter = DirectQwenPDFConverter()
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        # 첫 번째 비옵션 인수를 파일명으로 해석
        non_option_args = [a for a in sys.argv[1:] if not a.startswith('-')]
        specific_pdf = non_option_args[0] if non_option_args else None
        await converter.run(specific_pdf)
    else:
        await converter.run()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())