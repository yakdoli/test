"""
병렬 처리 엔진 구현
"""
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import psutil
from tqdm import tqdm

from modules.models.task_models import PDFTask, PageTask, TaskResult, BatchResult, TaskStatus
import config


class ResourceMonitor:
    """시스템 리소스 모니터링 클래스"""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 85.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self._monitoring = False
        self._stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }
    
    def get_current_usage(self) -> Dict[str, float]:
        """현재 시스템 리소스 사용량 반환"""
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }
    
    def should_throttle(self) -> bool:
        """리소스 사용량이 임계치를 초과했는지 확인"""
        usage = self.get_current_usage()
        return (usage['cpu'] > self.cpu_threshold or 
                usage['memory'] > self.memory_threshold)
    
    def get_optimal_worker_count(self) -> int:
        """최적 워커 수 계산"""
        cpu_count = mp.cpu_count()
        usage = self.get_current_usage()
        
        # 메모리 사용량에 따른 조정
        if usage['memory'] > 70:
            return max(1, cpu_count // 2)
        elif usage['memory'] > 50:
            return max(2, int(cpu_count * 0.75))
        else:
            return cpu_count


class ParallelProcessor:
    """병렬 처리 엔진"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.resource_monitor = ResourceMonitor()
        self.max_workers = max_workers or self.resource_monitor.get_optimal_worker_count()
        self._executor = None
        self._progress_callback: Optional[Callable] = None
        
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """진행 상황 콜백 설정"""
        self._progress_callback = callback
    
    def _update_progress(self, task_id: str, progress: float):
        """진행 상황 업데이트"""
        if self._progress_callback:
            self._progress_callback(task_id, progress)
    
    def process_pdf_batch(self, pdf_tasks: List[PDFTask], 
                         converter_func: Callable[[PDFTask], TaskResult]) -> BatchResult:
        """PDF 배치 병렬 처리"""
        print(f"🚀 {len(pdf_tasks)}개 PDF 병렬 처리 시작 (워커: {self.max_workers}개)")
        
        start_time = datetime.now()
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self._executor = executor
            
            # 작업 제출
            future_to_task = {
                executor.submit(converter_func, task): task 
                for task in pdf_tasks
            }
            
            # 진행 상황 추적
            with tqdm(total=len(pdf_tasks), desc="PDF 변환") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            pbar.set_postfix({"성공": f"{task.pdf_path.stem}"})
                        else:
                            pbar.set_postfix({"실패": f"{task.pdf_path.stem}"})
                            
                    except Exception as e:
                        error_result = TaskResult(
                            task_id=task.pdf_path.stem,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
                        pbar.set_postfix({"오류": f"{task.pdf_path.stem}"})
                    
                    pbar.update(1)
                    
                    # 리소스 모니터링 및 조정
                    if self.resource_monitor.should_throttle():
                        print("⚠️ 시스템 리소스 부족 - 처리 속도 조절 중...")
                        threading.Event().wait(2)  # 2초 대기
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        # 결과 집계
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchResult(
            total_tasks=len(pdf_tasks),
            successful_tasks=successful,
            failed_tasks=failed,
            skipped_tasks=0,
            total_processing_time=processing_time,
            results=results
        )
    
    def process_pages_parallel(self, page_tasks: List[PageTask],
                             converter_func: Callable[[PageTask], TaskResult]) -> List[TaskResult]:
        """페이지 단위 병렬 처리"""
        print(f"📄 {len(page_tasks)}개 페이지 병렬 처리 시작")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(converter_func, task): task 
                for task in page_tasks
            }
            
            with tqdm(total=len(page_tasks), desc="페이지 변환") as pbar:
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            pbar.set_postfix({"페이지": f"{task.page_number}"})
                        else:
                            pbar.set_postfix({"실패": f"{task.page_number}"})
                            
                    except Exception as e:
                        error_result = TaskResult(
                            task_id=task.task_id,
                            success=False,
                            error_message=str(e)
                        )
                        results.append(error_result)
                    
                    pbar.update(1)
        
        return results
    
    def shutdown(self):
        """처리기 종료"""
        if self._executor:
            self._executor.shutdown(wait=True)


class TaskManager:
    """작업 관리자 - 전체 워크플로우 조정"""
    
    def __init__(self, parallel_processor: ParallelProcessor):
        self.parallel_processor = parallel_processor
        self.checkpoint_manager = None  # 나중에 구현
        self.progress_tracker = None    # 나중에 구현
    
    def create_pdf_tasks(self, pdf_paths: List[Path]) -> List[PDFTask]:
        """PDF 경로 목록에서 작업 객체 생성"""
        tasks = []
        
        for pdf_path in pdf_paths:
            output_path = config.OUTPUT_DIR / f"{pdf_path.stem}.md"
            
            task = PDFTask(
                pdf_path=pdf_path,
                output_path=output_path,
                priority=0  # 기본 우선순위
            )
            tasks.append(task)
        
        return tasks
    
    def estimate_processing_time(self, tasks: List[PDFTask]) -> timedelta:
        """처리 시간 예상"""
        # 페이지당 평균 137초 (common.pdf 테스트 기준)
        avg_time_per_page = 137
        
        total_pages = 0
        for task in tasks:
            # PDF 페이지 수 추정 (실제로는 PDF를 열어서 확인해야 함)
            # 여기서는 파일 크기 기반 추정
            file_size_mb = task.pdf_path.stat().st_size / (1024 * 1024)
            estimated_pages = max(1, int(file_size_mb * 25))  # 대략적 추정
            total_pages += estimated_pages
        
        total_seconds = total_pages * avg_time_per_page / self.parallel_processor.max_workers
        return timedelta(seconds=total_seconds)
    
    def run_batch_processing(self, pdf_paths: List[Path]) -> BatchResult:
        """배치 처리 실행"""
        tasks = self.create_pdf_tasks(pdf_paths)
        
        # 처리 시간 예상
        estimated_time = self.estimate_processing_time(tasks)
        print(f"⏱️ 예상 처리 시간: {estimated_time}")
        
        # 실제 변환 함수는 나중에 구현
        def dummy_converter(task: PDFTask) -> TaskResult:
            """임시 변환 함수 - 실제 구현 필요"""
            return TaskResult(
                task_id=task.pdf_path.stem,
                success=True,
                output_path=task.output_path,
                processing_time=timedelta(seconds=10)
            )
        
        return self.parallel_processor.process_pdf_batch(tasks, dummy_converter)


if __name__ == "__main__":
    # 테스트 코드
    processor = ParallelProcessor(max_workers=2)
    manager = TaskManager(processor)
    
    # 테스트용 PDF 경로
    test_pdfs = list(config.PDF_DIR.glob("*.pdf"))[:2]  # 처음 2개만 테스트
    
    if test_pdfs:
        print("🧪 병렬 처리 테스트 시작")
        result = manager.run_batch_processing(test_pdfs)
        print(f"✅ 테스트 완료: {result.success_rate:.1%} 성공률")
    else:
        print("❌ 테스트할 PDF 파일이 없습니다.")