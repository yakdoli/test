"""
작업 관련 데이터 모델 정의
"""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any


class TaskStatus(Enum):
    """작업 상태 열거형"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingMode(Enum):
    """처리 모드 열거형"""
    FAST = "fast"
    QUALITY = "quality"
    SYNCFUSION = "syncfusion"


@dataclass
class PDFTask:
    """PDF 변환 작업 정보"""
    pdf_path: Path
    output_path: Path
    priority: int = 0
    estimated_time: Optional[timedelta] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PageTask:
    """페이지 변환 작업 정보"""
    image_path: Path
    page_number: int
    pdf_name: str
    status: TaskStatus = TaskStatus.NOT_STARTED
    retry_count: int = 0
    max_retries: int = 3
    processing_time: Optional[timedelta] = None
    error_message: Optional[str] = None
    
    @property
    def task_id(self) -> str:
        """고유 작업 ID 생성"""
        return f"{self.pdf_name}_page_{self.page_number:03d}"


@dataclass
class TaskResult:
    """작업 결과 정보"""
    task_id: str
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time: Optional[timedelta] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchResult:
    """배치 작업 결과"""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    total_processing_time: timedelta
    results: List[TaskResult]
    
    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    @property
    def failure_rate(self) -> float:
        """실패율 계산"""
        if self.total_tasks == 0:
            return 0.0
        return self.failed_tasks / self.total_tasks