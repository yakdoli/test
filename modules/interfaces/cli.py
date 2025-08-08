"""
고급 CLI 인터페이스 및 진행 상황 모니터링
tqdm와 비동기 모니터링을 결합한 사용자 인터페이스
"""
import argparse
import asyncio
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import psutil

from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# 상위 디렉토리의 config 모듈 import
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.progress_models import ProgressTracker, ResourceStats, ProgressLevel


class AsyncProgressMonitor:
    """비동기 진행 상황 모니터링"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.progress_bars: Dict[str, async_tqdm] = {}
        self.resource_monitor_task = None
        self.auto_save_task = None
        self.monitoring_active = False
        self.stats_history: List[ResourceStats] = []
        
    async def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        
        # 리소스 모니터링 태스크 시작
        self.resource_monitor_task = asyncio.create_task(self._monitor_resources())
        
        # 자동 저장 태스크 시작
        self.auto_save_task = asyncio.create_task(self._auto_save_progress())
        
        print("📊 비동기 모니터링 시작됨")
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        
        # 모든 진행률 바 닫기
        for pbar in self.progress_bars.values():
            pbar.close()
        self.progress_bars.clear()
        
        # 태스크들 취소
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        
        print("📊 비동기 모니터링 중지됨")
    
    async def _monitor_resources(self):
        """시스템 리소스 모니터링"""
        while self.monitoring_active:
            try:
                # CPU, 메모리, 디스크 사용량 수집
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                process = psutil.Process()
                
                stats = ResourceStats(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=(disk.used / disk.total) * 100,
                    available_memory_gb=memory.available / (1024**3),
                    process_memory_mb=process.memory_info().rss / (1024**2)
                )
                
                self.stats_history.append(stats)
                
                # 최근 100개 항목만 유지
                if len(self.stats_history) > 100:
                    self.stats_history.pop(0)
                
                # 리소스가 임계치를 초과하면 경고
                if stats.is_resource_critical():
                    print(f"⚠️ 리소스 사용량 높음: CPU {cpu_percent:.1f}%, RAM {memory.percent:.1f}%")
                
                await asyncio.sleep(5)  # 5초마다 체크
                
            except Exception as e:
                print(f"⚠️ 리소스 모니터링 오류: {e}")
                await asyncio.sleep(10)  # 오류 발생시 더 오래 대기
    
    async def _auto_save_progress(self):
        """자동 진행 상황 저장"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.checkpoint_manager.auto_save_interval)
                
                if self.checkpoint_manager.save_progress_checkpoint():
                    print("💾 진행 상황 자동 저장됨")
                
            except Exception as e:
                print(f"⚠️ 자동 저장 오류: {e}")
    
    def create_progress_bar(self, task_id: str, total: int, description: str) -> async_tqdm:
        """새 진행률 바 생성"""
        pbar = async_tqdm(
            total=total, 
            desc=description,
            unit='pages',
            colour='green',
            position=len(self.progress_bars),
            leave=True
        )
        
        self.progress_bars[task_id] = pbar
        return pbar
    
    def update_progress_bar(self, task_id: str, current: int, postfix: Dict[str, Any] = None):
        """진행률 바 업데이트"""
        if task_id in self.progress_bars:
            pbar = self.progress_bars[task_id]
            pbar.n = current
            if postfix:
                pbar.set_postfix(postfix)
            pbar.refresh()
    
    def complete_progress_bar(self, task_id: str):
        """진행률 바 완료"""
        if task_id in self.progress_bars:
            pbar = self.progress_bars[task_id]
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()
            del self.progress_bars[task_id]
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """리소스 사용량 요약"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-10:]  # 최근 10개 항목
        
        return {
            'current_cpu': recent_stats[-1].cpu_percent,
            'current_memory': recent_stats[-1].memory_percent,
            'current_process_memory_mb': recent_stats[-1].process_memory_mb,
            'avg_cpu': sum(s.cpu_percent for s in recent_stats) / len(recent_stats),
            'avg_memory': sum(s.memory_percent for s in recent_stats) / len(recent_stats),
            'peak_cpu': max(s.cpu_percent for s in recent_stats),
            'peak_memory': max(s.memory_percent for s in recent_stats),
            'samples_count': len(recent_stats)
        }


class CLIInterface:
    """명령행 인터페이스 클래스"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """argparse 파서 생성 및 설정"""
        parser = argparse.ArgumentParser(
            prog='pdf-converter',
            description='PDF 파일을 Markdown으로 변환하는 도구',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # 기본 옵션들
        parser.add_argument(
            'files',
            nargs='*',
            help='변환할 PDF 파일명 (확장자 제외). 지정하지 않으면 모든 PDF 파일을 처리합니다.'
        )
        
        parser.add_argument(
            '--list', '-l',
            action='store_true',
            help='사용 가능한 PDF 파일 목록을 표시합니다.'
        )
        
        parser.add_argument(
            '--mode', '-m',
            choices=['fast', 'quality', 'syncfusion'],
            default='syncfusion',
            help='처리 모드를 선택합니다. (기본값: syncfusion)'
        )
        
        parser.add_argument(
            '--parallel', '-p',
            type=int,
            metavar='N',
            help='병렬 처리할 작업 수를 지정합니다. (기본값: 시스템 자동 설정)'
        )
        
        parser.add_argument(
            '--resume', '-r',
            action='store_true',
            help='중단된 작업을 이어서 실행합니다.'
        )
        
        parser.add_argument(
            '--force', '-f',
            action='store_true',
            help='체크포인트를 무시하고 전체 작업을 다시 시작합니다.'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='상세한 디버그 정보를 표시합니다.'
        )
        
        parser.add_argument(
            '--stats', '-s',
            action='store_true',
            help='상세한 성능 통계를 파일로 저장합니다.'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=Path,
            metavar='DIR',
            help=f'출력 디렉토리를 지정합니다. (기본값: {config.OUTPUT_DIR})'
        )
        
        parser.add_argument(
            '--config', '-c',
            type=Path,
            metavar='FILE',
            help='설정 파일 경로를 지정합니다.'
        )
        
        return parser
    
    def _get_usage_examples(self) -> str:
        """사용 예시 문자열 반환"""
        return """
사용 예시:
  %(prog)s                          # 모든 PDF 파일 변환
  %(prog)s document1 document2      # 특정 PDF 파일들만 변환
  %(prog)s --list                   # 사용 가능한 PDF 파일 목록 표시
  %(prog)s --mode fast              # 빠른 처리 모드로 변환
  %(prog)s --parallel 4             # 4개 작업을 병렬로 처리
  %(prog)s --resume                 # 중단된 작업 이어서 실행
  %(prog)s --force                  # 처음부터 다시 시작
  %(prog)s --verbose --stats        # 상세 정보와 통계 저장
  %(prog)s --output ./results       # 출력 디렉토리 지정
        """
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        명령행 인수를 파싱합니다.
        
        Args:
            args: 파싱할 인수 리스트 (None이면 sys.argv 사용)
            
        Returns:
            argparse.Namespace: 파싱된 인수들
        """
        try:
            parsed_args = self.parser.parse_args(args)
            return parsed_args
        except SystemExit as e:
            # argparse가 --help나 오류 시 SystemExit을 발생시킴
            raise e
    
    def validate_arguments(self, args: argparse.Namespace) -> bool:
        """
        파싱된 인수들의 유효성을 검증합니다.
        
        Args:
            args: 검증할 인수들
            
        Returns:
            bool: 유효성 검증 결과
        """
        errors = []
        
        # PDF 디렉토리 존재 확인
        if not config.PDF_DIR.exists():
            errors.append(f"PDF 디렉토리가 존재하지 않습니다: {config.PDF_DIR}")
        
        # 특정 파일들이 지정된 경우 존재 확인
        if args.files:
            for filename in args.files:
                pdf_path = config.PDF_DIR / f"{filename}.pdf"
                if not pdf_path.exists():
                    errors.append(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # 병렬 처리 수 검증
        if args.parallel is not None:
            if args.parallel < 1:
                errors.append("병렬 처리 수는 1 이상이어야 합니다.")
            elif args.parallel > 16:
                errors.append("병렬 처리 수는 16 이하로 설정하는 것을 권장합니다.")
        
        # 출력 디렉토리 검증
        if args.output:
            try:
                args.output.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                errors.append(f"출력 디렉토리를 생성할 수 없습니다: {args.output} ({e})")
        
        # 설정 파일 검증
        if args.config and not args.config.exists():
            errors.append(f"설정 파일을 찾을 수 없습니다: {args.config}")
        
        # 상충하는 옵션 검증
        if args.resume and args.force:
            errors.append("--resume과 --force 옵션은 동시에 사용할 수 없습니다.")
        
        # 오류가 있으면 출력하고 False 반환
        if errors:
            print("❌ 인수 검증 오류:", file=sys.stderr)
            for error in errors:
                print(f"   • {error}", file=sys.stderr)
            return False
        
        return True
    
    def display_help(self) -> None:
        """도움말을 표시합니다."""
        self.parser.print_help()
        
        # 추가 도움말 정보
        print("\n" + "=" * 60)
        print("📖 상세 도움말")
        print("=" * 60)
        
        print("\n🔧 처리 모드 설명:")
        print("  fast      - 빠른 처리 (낮은 품질, 높은 속도)")
        print("  quality   - 고품질 처리 (높은 품질, 낮은 속도)")
        print("  syncfusion- Syncfusion SDK 매뉴얼 최적화 모드")
        
        print("\n⚡ 성능 최적화 팁:")
        print("  • --parallel 옵션으로 병렬 처리 수를 조절하세요")
        print("  • 대용량 파일은 --mode fast로 먼저 테스트해보세요")
        print("  • --resume 옵션으로 중단된 작업을 이어서 실행하세요")
        print("  • --stats 옵션으로 성능 분석 정보를 확인하세요")
        
        print("\n🔍 문제 해결:")
        print("  • Ollama 서버가 실행 중인지 확인: ollama serve")
        print("  • 모델이 설치되어 있는지 확인: ollama list")
        print("  • 충분한 디스크 공간이 있는지 확인하세요")
        print("  • --verbose 옵션으로 상세 로그를 확인하세요")
        
        print("\n📁 디렉토리 구조:")
        print(f"  PDF 입력:  {config.PDF_DIR}")
        print(f"  임시 파일: {config.STAGING_DIR}")
        print(f"  결과 출력: {config.OUTPUT_DIR}")
    
    def list_available_pdfs(self, detailed: bool = False) -> None:
        """
        사용 가능한 PDF 파일 목록을 표시합니다.
        
        Args:
            detailed: 상세 정보 표시 여부
        """
        print("📋 사용 가능한 PDF 파일:")
        print("=" * 50)
        
        if not config.PDF_DIR.exists():
            print(f"❌ PDF 디렉토리가 존재하지 않습니다: {config.PDF_DIR}")
            print(f"💡 다음 명령으로 디렉토리를 생성하세요: mkdir -p {config.PDF_DIR}")
            return
        
        pdf_files = list(config.PDF_DIR.glob("*.pdf"))
        
        if not pdf_files:
            print(f"📁 {config.PDF_DIR}에 PDF 파일이 없습니다.")
            print(f"💡 PDF 파일을 {config.PDF_DIR} 디렉토리에 복사하세요.")
            return
        
        # 통계 정보 수집
        total_size = 0
        converted_count = 0
        pending_count = 0
        
        # 파일 정보와 함께 목록 표시
        for i, pdf_file in enumerate(sorted(pdf_files), 1):
            file_size = pdf_file.stat().st_size
            size_mb = file_size / (1024 * 1024)
            total_size += file_size
            
            # 파일명 (확장자 제외)
            filename = pdf_file.stem
            
            # 변환 상태 확인
            output_file = config.OUTPUT_DIR / f"{filename}.md"
            staging_dir = config.STAGING_DIR / filename
            
            if output_file.exists():
                status = "✅ 변환 완료"
                converted_count += 1
            elif staging_dir.exists() and list(staging_dir.glob("*.jpeg")):
                status = "🔄 이미지 변환 완료"
                pending_count += 1
            else:
                status = "⏳ 변환 대기"
                pending_count += 1
            
            print(f"{i:2d}. {filename}")
            print(f"    📄 파일: {pdf_file.name}")
            print(f"    📊 크기: {size_mb:.1f} MB")
            print(f"    🔄 상태: {status}")
            
            if detailed:
                # 상세 정보 표시
                if output_file.exists():
                    output_size = output_file.stat().st_size / 1024  # KB
                    print(f"    📝 출력: {output_file.name} ({output_size:.1f} KB)")
                
                if staging_dir.exists():
                    image_count = len(list(staging_dir.glob("*.jpeg")))
                    if image_count > 0:
                        print(f"    🖼️  이미지: {image_count}개 페이지")
            
            print()
        
        # 요약 정보
        total_size_mb = total_size / (1024 * 1024)
        print("📊 요약 정보:")
        print(f"   총 파일 수: {len(pdf_files)}개")
        print(f"   총 크기: {total_size_mb:.1f} MB")
        print(f"   변환 완료: {converted_count}개")
        print(f"   변환 대기: {pending_count}개")
        
        print(f"\n📁 디렉토리 정보:")
        print(f"   PDF 입력: {config.PDF_DIR}")
        print(f"   임시 파일: {config.STAGING_DIR}")
        print(f"   결과 출력: {config.OUTPUT_DIR}")
        
        if pending_count > 0:
            print(f"\n💡 변환 시작하려면: python main.py")
            print(f"💡 특정 파일만: python main.py {pdf_files[0].stem}")


class CLIError(Exception):
    """CLI 관련 오류"""
    pass


class ArgumentValidationError(CLIError):
    """인수 검증 오류"""
    pass


class CLIErrorHandler:
    """CLI 오류 처리 클래스"""
    
    @staticmethod
    def handle_parsing_error(error: Exception) -> None:
        """
        파싱 오류를 처리합니다.
        
        Args:
            error: 발생한 오류
        """
        if isinstance(error, SystemExit):
            # argparse의 정상적인 종료 (--help 등)
            raise error
        else:
            print(f"❌ 명령행 인수 파싱 오류: {error}", file=sys.stderr)
            print("도움말을 보려면 --help 옵션을 사용하세요.", file=sys.stderr)
            sys.exit(1)
    
    @staticmethod
    def handle_validation_error(errors: List[str]) -> None:
        """
        검증 오류를 처리합니다.
        
        Args:
            errors: 오류 메시지 리스트
        """
        print("❌ 인수 검증 실패:", file=sys.stderr)
        for error in errors:
            print(f"   • {error}", file=sys.stderr)
        print("\n도움말을 보려면 --help 옵션을 사용하세요.", file=sys.stderr)
        sys.exit(1)
    
    @staticmethod
    def handle_file_not_found_error(filename: str) -> None:
        """
        파일을 찾을 수 없는 오류를 처리합니다.
        
        Args:
            filename: 찾을 수 없는 파일명
        """
        print(f"❌ 파일을 찾을 수 없습니다: {filename}", file=sys.stderr)
        print("사용 가능한 파일 목록을 보려면 --list 옵션을 사용하세요.", file=sys.stderr)
        sys.exit(1)


class EnhancedCLI:
    """향상된 CLI 인터페이스"""
    
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
        self.progress_monitor = AsyncProgressMonitor(self.checkpoint_manager)
        self.cli_interface = CLIInterface()
        self.interruption_handler_set = False
        
    def setup_interruption_handler(self):
        """중단 신호 핸들러 설정"""
        import signal
        
        def signal_handler(signum, frame):
            print(f"\n🛑 중단 신호 수신됨 (신호: {signum})")
            asyncio.create_task(self.handle_interruption())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self.interruption_handler_set = True
        print("🛡️ 중단 신호 핸들러 설정됨")
    
    async def handle_interruption(self):
        """중단 처리"""
        print("💾 중단 감지 - 진행 상황 저장 중...")
        
        try:
            # 현재 진행 상황 저장
            if self.checkpoint_manager.save_progress_checkpoint():
                print("✅ 진행 상황 저장 완료")
            else:
                print("❌ 진행 상황 저장 실패")
            
            # 모니터링 중지
            await self.progress_monitor.stop_monitoring()
            
            print("🏁 안전하게 중단됨. 나중에 --resume 옵션으로 재시작할 수 있습니다.")
            
        except Exception as e:
            print(f"❌ 중단 처리 중 오류: {e}")
        
        # 프로그램 종료
        import sys
        sys.exit(0)
    
    def print_session_info(self, session_id: str):
        """세션 정보 출력"""
        summary = self.checkpoint_manager.get_progress_summary()
        if summary:
            print(f"\n📋 세션 정보:")
            print(f"   세션 ID: {session_id}")
            print(f"   시작 시간: {summary['started_at']}")
            print(f"   전체 작업: {summary['total_tasks']}")
            print(f"   완료: {summary['completed_tasks']}")
            print(f"   실패: {summary['failed_tasks']}")
            print(f"   진행률: {summary['progress_percent']:.1f}%")
            print(f"   경과 시간: {summary['elapsed_time']}")
    
    def print_chunk_summary(self, pdf_name: Optional[str] = None):
        """청크 처리 요약 출력"""
        summary = self.checkpoint_manager.get_chunk_progress_summary(pdf_name)
        if summary:
            print(f"\n📦 청크 처리 요약 ({'전체' if not pdf_name else pdf_name}):")
            print(f"   총 청크: {summary['total_chunks']}")
            print(f"   완료된 청크: {summary['completed_chunks']}")
            print(f"   실패한 청크: {summary['failed_chunks']}")
            print(f"   진행 중인 청크: {summary['in_progress_chunks']}")
            print(f"   청크 진행률: {summary['chunk_progress_percent']:.1f}%")
            print(f"   페이지 진행률: {summary['page_progress_percent']:.1f}%")
            print(f"   처리된 페이지: {summary['processed_pages']}/{summary['total_pages']}")
    
    def print_resource_summary(self):
        """리소스 사용량 요약 출력"""
        resource_summary = self.progress_monitor.get_resource_summary()
        if resource_summary:
            print(f"\n💻 시스템 리소스:")
            print(f"   현재 CPU: {resource_summary['current_cpu']:.1f}%")
            print(f"   현재 메모리: {resource_summary['current_memory']:.1f}%")
            print(f"   프로세스 메모리: {resource_summary['current_process_memory_mb']:.1f}MB")
            print(f"   평균 CPU: {resource_summary['avg_cpu']:.1f}%")
            print(f"   평균 메모리: {resource_summary['avg_memory']:.1f}%")
            print(f"   피크 CPU: {resource_summary['peak_cpu']:.1f}%")
            print(f"   피크 메모리: {resource_summary['peak_memory']:.1f}%")
    
    async def run_with_monitoring(self, task_func: Callable, *args, **kwargs):
        """모니터링과 함께 작업 실행"""
        if not self.interruption_handler_set:
            self.setup_interruption_handler()
        
        try:
            # 모니터링 시작
            await self.progress_monitor.start_monitoring()
            
            # 실제 작업 실행
            result = await task_func(*args, **kwargs)
            
            return result
            
        except KeyboardInterrupt:
            print("\n🛑 키보드 인터럽트 감지")
            await self.handle_interruption()
            
        except Exception as e:
            print(f"❌ 작업 실행 중 오류: {e}")
            raise
            
        finally:
            # 모니터링 중지
            await self.progress_monitor.stop_monitoring()
            
            # 최종 요약 출력
            self.print_final_summary()
    
    def print_final_summary(self):
        """최종 처리 요약 출력"""
        print(f"\n{'='*60}")
        print("🎯 최종 처리 요약")
        print(f"{'='*60}")
        
        session_id = self.checkpoint_manager.current_session_id
        if session_id:
            self.print_session_info(session_id)
            self.print_chunk_summary()
            self.print_resource_summary()
        
        print(f"{'='*60}")
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """명령행 인수 파싱 (CLI 인터페이스 위임)"""
        return self.cli_interface.parse_arguments(args)
    
    def validate_arguments(self, args: argparse.Namespace) -> bool:
        """인수 유효성 검증 (CLI 인터페이스 위임)"""
        return self.cli_interface.validate_arguments(args)
    
    def list_available_pdfs(self, detailed: bool = False):
        """PDF 파일 목록 출력 (CLI 인터페이스 위임)"""
        self.cli_interface.list_available_pdfs(detailed)


def create_cli_interface() -> CLIInterface:
    """CLI 인터페이스 인스턴스를 생성합니다."""
    return CLIInterface()


def create_enhanced_cli() -> EnhancedCLI:
    """향상된 CLI 인스턴스 생성"""
    return EnhancedCLI()


def main():
    """CLI 모듈 테스트용 메인 함수"""
    cli = CLIInterface()
    
    print("🧪 CLI 인터페이스 테스트")
    print("=" * 40)
    
    # 기본 파싱 테스트
    try:
        args = cli.parse_arguments(['--list'])
        print("✅ 기본 파싱 테스트 성공")
        
        if args.list:
            cli.list_available_pdfs()
            
    except SystemExit:
        print("✅ SystemExit 처리 정상")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    main()