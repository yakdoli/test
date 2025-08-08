"""
진행 상황 추적 관련 데이터 모델
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class ProgressLevel(Enum):
    """진행 상황 레벨"""
    OVERALL = "overall"      # 전체 진행률
    PDF = "pdf"             # PDF별 진행률
    PAGE = "page"           # 페이지별 진행률


@dataclass
class ProgressInfo:
    """진행 상황 정보"""
    level: ProgressLevel
    task_id: str
    current: int
    total: int
    start_time: datetime
    last_update: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """진행률 (백분율)"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        """경과 시간"""
        return self.last_update - self.start_time
    
    @property
    def eta(self) -> Optional[timedelta]:
        """예상 완료 시간 (Estimated Time of Arrival)"""
        if self.current == 0 or self.total == 0:
            return None
        
        elapsed = self.elapsed_time
        rate = self.current / elapsed.total_seconds()  # 초당 처리량
        remaining_items = self.total - self.current
        
        if rate > 0:
            eta_seconds = remaining_items / rate
            return timedelta(seconds=eta_seconds)
        
        return None
    
    def update_progress(self, current: int) -> None:
        """진행 상황 업데이트"""
        self.current = current
        self.last_update = datetime.now()
        
        # ETA 계산
        eta = self.eta
        if eta:
            self.estimated_completion = self.last_update + eta


@dataclass
class ResourceStats:
    """시스템 리소스 사용량 통계"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    available_memory_gb: float
    process_memory_mb: float
    
    def is_resource_critical(self, cpu_threshold: float = 90.0, 
                           memory_threshold: float = 85.0) -> bool:
        """리소스 사용량이 임계치를 초과했는지 확인"""
        return (self.cpu_percent > cpu_threshold or 
                self.memory_percent > memory_threshold)


@dataclass
class PerformanceMetrics:
    """성능 지표"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_pages_processed: int = 0
    total_processing_time: timedelta = field(default_factory=lambda: timedelta())
    average_page_time: Optional[timedelta] = None
    peak_memory_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    resource_stats: List[ResourceStats] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def pages_per_minute(self) -> float:
        """분당 페이지 처리량"""
        if self.total_processing_time.total_seconds() == 0:
            return 0.0
        
        minutes = self.total_processing_time.total_seconds() / 60
        return self.total_pages_processed / minutes
    
    def add_resource_stats(self, stats: ResourceStats) -> None:
        """리소스 통계 추가"""
        self.resource_stats.append(stats)
        
        # 피크 사용량 업데이트
        self.peak_cpu_usage = max(self.peak_cpu_usage, stats.cpu_percent)
        self.peak_memory_usage = max(self.peak_memory_usage, stats.memory_percent)
    
    def calculate_average_page_time(self) -> None:
        """평균 페이지 처리 시간 계산"""
        if self.total_pages_processed > 0:
            total_seconds = self.total_processing_time.total_seconds()
            avg_seconds = total_seconds / self.total_pages_processed
            self.average_page_time = timedelta(seconds=avg_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (저장용)"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_pages_processed': self.total_pages_processed,
            'total_processing_time_seconds': self.total_processing_time.total_seconds(),
            'average_page_time_seconds': self.average_page_time.total_seconds() if self.average_page_time else None,
            'success_rate': self.success_rate,
            'pages_per_minute': self.pages_per_minute,
            'peak_memory_usage': self.peak_memory_usage,
            'peak_cpu_usage': self.peak_cpu_usage,
            'resource_stats_count': len(self.resource_stats),
            'error_count': len(self.error_messages)
        }


@dataclass
class ProgressTracker:
    """진행 상황 추적기"""
    session_id: str
    progress_info: Dict[str, ProgressInfo] = field(default_factory=dict)
    performance_metrics: Optional[PerformanceMetrics] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = PerformanceMetrics(
                session_id=self.session_id,
                start_time=datetime.now()
            )
    
    def start_tracking(self, task_id: str, total_items: int, 
                      level: ProgressLevel = ProgressLevel.PDF) -> None:
        """진행 상황 추적 시작"""
        self.progress_info[task_id] = ProgressInfo(
            level=level,
            task_id=task_id,
            current=0,
            total=total_items,
            start_time=datetime.now()
        )
    
    def update_progress(self, task_id: str, current: int) -> None:
        """진행 상황 업데이트"""
        if task_id in self.progress_info:
            self.progress_info[task_id].update_progress(current)
    
    def complete_task(self, task_id: str, success: bool = True) -> None:
        """작업 완료 처리"""
        if task_id in self.progress_info:
            progress = self.progress_info[task_id]
            progress.current = progress.total
            progress.last_update = datetime.now()
            
            # 성능 지표 업데이트
            if self.performance_metrics:
                if success:
                    self.performance_metrics.completed_tasks += 1
                else:
                    self.performance_metrics.failed_tasks += 1
    
    def get_overall_progress(self) -> Optional[ProgressInfo]:
        """전체 진행률 계산"""
        if not self.progress_info:
            return None
        
        total_items = sum(p.total for p in self.progress_info.values())
        current_items = sum(p.current for p in self.progress_info.values())
        
        if total_items == 0:
            return None
        
        # 가장 빠른 시작 시간 찾기
        start_times = [p.start_time for p in self.progress_info.values()]
        earliest_start = min(start_times) if start_times else datetime.now()
        
        overall = ProgressInfo(
            level=ProgressLevel.OVERALL,
            task_id="overall",
            current=current_items,
            total=total_items,
            start_time=earliest_start
        )
        
        return overall
    
    def get_eta_summary(self) -> Dict[str, Any]:
        """ETA 요약 정보 반환"""
        overall = self.get_overall_progress()
        if not overall:
            return {}
        
        eta = overall.eta
        return {
            'progress_percent': overall.progress_percent,
            'elapsed_time': str(overall.elapsed_time),
            'eta': str(eta) if eta else "계산 중...",
            'estimated_completion': overall.estimated_completion.strftime('%Y-%m-%d %H:%M:%S') if overall.estimated_completion else None
        }