"""
Enhanced PDF to Markdown 변환기 메인 프로그램
모든 개선 사항이 통합된 버전
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Optional
import config

# 모듈 import
from pdf_converter import PDFConverter
from unified_ollama_client import UnifiedVLClient
from modules.interfaces.cli import create_enhanced_cli
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.task_models import PDFTask, TaskStatus
from git_automation import create_git_automation


class EnhancedPDFToMarkdownConverter:
    """향상된 PDF to Markdown 변환기"""
    
    def __init__(self):
        self.pdf_converter = PDFConverter()
        self.vl_client = UnifiedVLClient()
        self.checkpoint_manager = CheckpointManager()
        self.git_automation = create_git_automation()
        self.cli = create_enhanced_cli()
        self.output_dir = config.OUTPUT_DIR
        self.git_enabled = False
        
    async def initialize(self) -> bool:
        """시스템 초기화"""
        print("🚀 Enhanced PDF to Markdown 변환기 초기화 중...")
        print("=" * 60)
        
        # VL 클라이언트 초기화
        print("🧠 비전-언어 모델 초기화 중...")
        if not await self.vl_client.initialize():
            print("❌ 모델 초기화 실패")
            return False
        
        if not await self.vl_client.check_availability():
            print("❌ 모델 사용 불가능")
            return False
        
        # 모델 정보 출력
        mode = "Direct Qwen2.5-VL-7B" if config.USE_DIRECT_QWEN else "Xinference API"
        print(f"✅ 모델 준비 완료: {mode}")
        
        # Git 상태 확인
        try:
            self.git_automation.print_status()
            self.git_enabled = True
            print("✅ Git 자동화 준비 완료")
        except Exception as e:
            print(f"⚠️ Git 자동화 비활성화: {e}")
            self.git_enabled = False
        
        print("✅ 시스템 초기화 완료")
        return True
    
    async def convert_single_pdf_enhanced(self, pdf_name: str, image_paths: List[Path]) -> bool:
        """단일 PDF 향상된 변환"""
        print(f"\n📄 '{pdf_name}' 향상된 변환 시작...")
        
        try:
            # 청크 상태 생성
            total_pages = len(image_paths)
            chunk_size = getattr(config, 'CHUNK_SIZE', 3)
            
            chunk_ids = self.checkpoint_manager.create_chunk_states(
                pdf_name, total_pages, chunk_size
            )
            
            # 진행 상황 추적 시작
            if self.checkpoint_manager.progress_tracker:
                self.checkpoint_manager.progress_tracker.start_tracking(
                    pdf_name, total_pages
                )
            
            # 실제 변환 실행
            start_time = asyncio.get_event_loop().time()
            
            # 청크별 처리 상태 추적
            for chunk_id in chunk_ids:
                self.checkpoint_manager.update_chunk_status(
                    chunk_id, TaskStatus.IN_PROGRESS
                )
            
            # 비동기 병렬 변환 실행
            markdown_content = await self.vl_client.convert_images_to_markdown_parallel(image_paths)
            
            if not markdown_content.strip():
                print(f"❌ '{pdf_name}' 변환 실패: 빈 내용")
                # 모든 청크를 실패로 표시
                for chunk_id in chunk_ids:
                    self.checkpoint_manager.update_chunk_status(
                        chunk_id, TaskStatus.FAILED, error_message="Empty content"
                    )
                return False
            
            # Syncfusion 특화 후처리
            if config.SYNCFUSION_MODE:
                markdown_content = self.vl_client.post_process_syncfusion_content(
                    markdown_content, pdf_name
                )
            
            # 마크다운 파일 저장
            output_file = self.output_dir / f"{pdf_name}_enhanced.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # 모든 청크를 완료로 표시
            for chunk_id in chunk_ids:
                # 청크에 해당하는 페이지들을 처리됨으로 표시
                chunk_state = self.checkpoint_manager.load_state().chunk_states[chunk_id]
                processed_pages = list(range(chunk_state.page_start, chunk_state.page_end + 1))
                
                self.checkpoint_manager.update_chunk_status(
                    chunk_id, TaskStatus.COMPLETED, processed_pages=processed_pages
                )
            
            # 전체 작업 완료
            self.checkpoint_manager.update_task_status(pdf_name, TaskStatus.COMPLETED)
            
            # 처리 시간 계산
            processing_time = asyncio.get_event_loop().time() - start_time
            
            print(f"✅ '{pdf_name}' 향상된 변환 완료!")
            print(f"📁 출력 파일: {output_file}")
            print(f"⏱️ 처리 시간: {processing_time:.1f}초")
            print(f"📊 페이지 수: {total_pages}")
            print(f"📦 청크 수: {len(chunk_ids)}")
            
            # 성능 통계 출력
            stats = self.vl_client.get_performance_stats()
            print(f"🔧 처리 모드: {stats.get('mode', 'unknown')}")
            
            if 'detailed_stats' in stats:
                detailed = stats['detailed_stats']
                avg_time = detailed.get('average_processing_time', 0)
                if avg_time > 0:
                    print(f"⚡ 평균 처리 시간: {avg_time:.1f}초/페이지")
            
            return True
            
        except Exception as e:
            print(f"❌ '{pdf_name}' 변환 실패: {str(e)}")
            
            # 모든 청크를 실패로 표시
            for chunk_id in chunk_ids:
                self.checkpoint_manager.update_chunk_status(
                    chunk_id, TaskStatus.FAILED, error_message=str(e)
                )
            
            self.checkpoint_manager.update_task_status(
                pdf_name, TaskStatus.FAILED, error_message=str(e)
            )
            
            return False
    
    async def run_enhanced(self, specific_pdf: str = None, resume: bool = False):
        """향상된 변환 프로세스 실행"""
        print("🚀 Enhanced PDF to Markdown 변환기 시작")
        print("=" * 60)
        
        # 재시작 모드 확인
        if resume:
            resumable_tasks = self.checkpoint_manager.get_resumable_tasks()
            if not resumable_tasks:
                print("📋 재시작할 작업이 없습니다.")
                return
            print(f"🔄 재시작 모드: {len(resumable_tasks)}개 작업 재개")
        
        # PDF 이미지 변환 (1단계)
        print("\n📸 1단계: PDF → 이미지 변환 (향상된 체크포인트)")
        pdf_images = self.pdf_converter.convert_pdfs(specific_pdf)
        
        if not pdf_images:
            print("❌ 처리할 PDF가 없습니다.")
            return
        
        # 작업 목록 생성 및 세션 시작
        tasks = []
        for pdf_name in pdf_images.keys():
            task = PDFTask(
                pdf_path=config.PDF_DIR / f"{pdf_name}.pdf",
                output_path=self.output_dir / f"{pdf_name}_enhanced.md"
            )
            tasks.append(task)
        
        if not resume:
            session_id = self.checkpoint_manager.create_new_session(tasks)
        else:
            session_id = self.checkpoint_manager.current_session_id or "resumed"
        
        print(f"\n📝 2단계: 향상된 마크다운 변환 ({len(pdf_images)}개 PDF)")
        print(f"🆔 세션 ID: {session_id}")
        
        # 향상된 변환 실행 (비동기 모니터링과 함께)
        async def conversion_task():
            success_count = 0
            total_count = len(pdf_images)
            
            for pdf_name, image_paths in pdf_images.items():
                # 재시작 모드에서 이미 완료된 작업 건너뛰기
                if resume:
                    completed_tasks = self.checkpoint_manager.get_completed_tasks()
                    if pdf_name in completed_tasks:
                        print(f"⏭️ '{pdf_name}' 이미 완료됨 - 건너뛰기")
                        success_count += 1
                        continue
                
                if await self.convert_single_pdf_enhanced(pdf_name, image_paths):
                    success_count += 1
                    
                    # Git 커밋 (활성화된 경우)
                    if self.git_enabled:
                        await self._commit_pdf_conversion(pdf_name)
            
            return success_count, total_count
        
        # 모니터링과 함께 실행
        success_count, total_count = await self.cli.run_with_monitoring(conversion_task)
        
        # 최종 Git 커밋
        if self.git_enabled and success_count > 0:
            await self._commit_session_completion(session_id, success_count, total_count)
        
        return success_count, total_count
    
    async def _commit_pdf_conversion(self, pdf_name: str):
        """개별 PDF 변환 완료 시 커밋"""
        try:
            task_name = f"Convert {pdf_name} to Markdown"
            task_description = f"Enhanced PDF to Markdown conversion completed for {pdf_name}"
            
            # 관련 파일들 확인
            related_files = [
                f"output/{pdf_name}_enhanced.md",
                f"staging/{pdf_name}/"
            ]
            
            # 존재하는 파일만 필터링
            existing_files = []
            for file_pattern in related_files:
                file_path = config.BASE_DIR / file_pattern
                if file_path.exists():
                    existing_files.append(str(file_path.relative_to(config.BASE_DIR)))
            
            success = self.git_automation.create_task_commit(
                task_name, task_description, existing_files
            )
            
            if success:
                print(f"📝 Git 커밋 완료: {pdf_name}")
            
        except Exception as e:
            print(f"⚠️ Git 커밋 실패 ({pdf_name}): {e}")
    
    async def _commit_session_completion(self, session_id: str, success_count: int, total_count: int):
        """세션 완료 시 최종 커밋"""
        try:
            task_name = f"Complete conversion session {session_id[:8]}"
            task_description = f"""PDF to Markdown conversion session completed
            
Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)
Session ID: {session_id}
Processing mode: {'Direct Qwen2.5-VL' if config.USE_DIRECT_QWEN else 'Xinference API'}
Enhanced features: DPI-aware checkpointing, chunk-based processing, async monitoring"""
            
            success = self.git_automation.commit_and_push_task(
                task_name, task_description,
                create_branch=False,
                push_to_remote=True
            )
            
            if success:
                print(f"🚀 세션 완료 커밋 및 푸시 완료")
            
        except Exception as e:
            print(f"⚠️ 세션 완료 커밋 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self.vl_client, 'cleanup'):
            self.vl_client.cleanup()
        
        print("🧹 리소스 정리 완료")


async def main():
    """메인 함수"""
    converter = EnhancedPDFToMarkdownConverter()
    
    # CLI 인수 파싱
    try:
        args = converter.cli.parse_arguments()
        if not converter.cli.validate_arguments(args):
            return
    except SystemExit:
        return
    
    # CLI로 전달된 Xinference Base URL 적용 (가능한 한 이른 시점에)
    if hasattr(args, 'xinference_base_url') and args.xinference_base_url:
        new_url = args.xinference_base_url.strip()
        try:
            prev = getattr(config, 'XINFERENCE_BASE_URL', None)
            config.XINFERENCE_BASE_URL = new_url
            import os as _os
            _os.environ['XINFERENCE_BASE_URL'] = config.XINFERENCE_BASE_URL
            print(f"🌐 Xinference Base URL 적용: {prev} -> {config.XINFERENCE_BASE_URL}")
        except Exception as e:
            print(f"⚠️ Xinference Base URL 적용 실패: {e}")

    # 도움말 또는 목록 표시
    if hasattr(args, 'list') and args.list:
        converter.cli.list_available_pdfs(detailed=True)
        return
    
    try:
        # 시스템 초기화
        if not await converter.initialize():
            print("❌ 시스템 초기화 실패")
            return
        
        # 변환 실행
        specific_pdf = args.files[0] if args.files else None
        resume = getattr(args, 'resume', False)
        
        success_count, total_count = await converter.run_enhanced(
            specific_pdf=specific_pdf,
            resume=resume
        )
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("🎉 Enhanced 변환 완료!")
        print(f"✅ 성공: {success_count}/{total_count}")
        
        if success_count < total_count:
            print(f"❌ 실패: {total_count - success_count}")
            print("💡 --resume 옵션으로 실패한 작업을 재시작할 수 있습니다.")
        
        print(f"📁 출력 디렉토리: {converter.output_dir}")
        
    except KeyboardInterrupt:
        print("\n🛑 사용자 중단 감지됨")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        converter.cleanup()


if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())