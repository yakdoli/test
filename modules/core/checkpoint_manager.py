"""
체크포인트 관리 시스템
작업 진행 상태를 저장하고 복원하는 기능을 제공합니다.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict, dataclass

from modules.models.task_models import PDFTask, TaskStatus, TaskResult
from modules.models.progress_models import ProgressTracker, ProgressLevel, PerformanceMetrics
import config


@dataclass
class ChunkState:
    """청크 처리 상태 정보"""
    chunk_id: str
    pdf_name: str
    chunk_index: int
    total_chunks: int
    page_start: int
    page_end: int
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processed_pages: List[int] = None
    
    def __post_init__(self):
        if self.processed_pages is None:
            self.processed_pages = []

@dataclass
class ProcessingState:
    """전체 처리 상태 정보"""
    session_id: str
    started_at: datetime
    last_updated: datetime
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_states: Dict[str, Dict[str, Any]]  # task_id -> task_state
    chunk_states: Dict[str, ChunkState]  # chunk_id -> chunk_state
    configuration: Dict[str, Any]
    progress_tracker: Optional[Dict[str, Any]] = None  # ProgressTracker 직렬화 데이터
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        # ChunkState 객체들을 딕셔너리로 변환
        chunk_states_dict = {}
        for chunk_id, chunk_state in self.chunk_states.items():
            chunk_states_dict[chunk_id] = {
                'chunk_id': chunk_state.chunk_id,
                'pdf_name': chunk_state.pdf_name,
                'chunk_index': chunk_state.chunk_index,
                'total_chunks': chunk_state.total_chunks,
                'page_start': chunk_state.page_start,
                'page_end': chunk_state.page_end,
                'status': chunk_state.status.value,
                'started_at': chunk_state.started_at.isoformat() if chunk_state.started_at else None,
                'completed_at': chunk_state.completed_at.isoformat() if chunk_state.completed_at else None,
                'error_message': chunk_state.error_message,
                'processed_pages': chunk_state.processed_pages
            }
        
        return {
            'session_id': self.session_id,
            'started_at': self.started_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'task_states': self.task_states,
            'chunk_states': chunk_states_dict,
            'configuration': self.configuration,
            'progress_tracker': self.progress_tracker
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """딕셔너리에서 복원"""
        # ChunkState 객체들을 복원
        chunk_states = {}
        chunk_states_data = data.get('chunk_states', {})
        for chunk_id, chunk_data in chunk_states_data.items():
            chunk_states[chunk_id] = ChunkState(
                chunk_id=chunk_data['chunk_id'],
                pdf_name=chunk_data['pdf_name'],
                chunk_index=chunk_data['chunk_index'],
                total_chunks=chunk_data['total_chunks'],
                page_start=chunk_data['page_start'],
                page_end=chunk_data['page_end'],
                status=TaskStatus(chunk_data['status']),
                started_at=datetime.fromisoformat(chunk_data['started_at']) if chunk_data['started_at'] else None,
                completed_at=datetime.fromisoformat(chunk_data['completed_at']) if chunk_data['completed_at'] else None,
                error_message=chunk_data.get('error_message'),
                processed_pages=chunk_data.get('processed_pages', [])
            )
        
        return cls(
            session_id=data['session_id'],
            started_at=datetime.fromisoformat(data['started_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            total_tasks=data['total_tasks'],
            completed_tasks=data['completed_tasks'],
            failed_tasks=data['failed_tasks'],
            task_states=data['task_states'],
            chunk_states=chunk_states,
            configuration=data['configuration'],
            progress_tracker=data.get('progress_tracker')
        )


class CheckpointManager:
    """체크포인트 관리자"""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or (config.BASE_DIR / ".checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.state_file = self.checkpoint_dir / "processing_state.json"
        self.backup_file = self.checkpoint_dir / "processing_state.backup.json"
        self.progress_tracker: Optional[ProgressTracker] = None
        self.auto_save_interval = 30  # 30초마다 자동 저장
    
    def generate_session_id(self) -> str:
        """새 세션 ID 생성"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def create_new_session(self, tasks: List[PDFTask]) -> str:
        """새 처리 세션 생성"""
        session_id = self.generate_session_id()
        self.current_session_id = session_id
        
        # 설정 정보 수집
        configuration = {
            'dpi': config.DPI,
            'image_format': config.IMAGE_FORMAT,
            'ollama_model': getattr(config, 'XINFERENCE_MODEL_NAME', 'qwen2-vl-instruct'),
            'syncfusion_mode': config.SYNCFUSION_MODE,
            'extract_code_snippets': config.EXTRACT_CODE_SNIPPETS,
            'parallel_processing': True,
            'use_direct_qwen': getattr(config, 'USE_DIRECT_QWEN', False),
            'chunk_size': getattr(config, 'CHUNK_SIZE', 3),
            'max_concurrent': getattr(config, 'MAX_CONCURRENT_REQUESTS', 12)
        }
        
        # 초기 상태 생성
        task_states = {}
        for task in tasks:
            task_id = task.pdf_path.stem
            task_states[task_id] = {
                'pdf_path': str(task.pdf_path),
                'output_path': str(task.output_path),
                'status': TaskStatus.NOT_STARTED.value,
                'created_at': task.created_at.isoformat() if task.created_at else None,
                'started_at': None,
                'completed_at': None,
                'error_message': None,
                'retry_count': 0
            }
        
        # 진행 상황 추적기 초기화
        self.progress_tracker = ProgressTracker(session_id=session_id)
        
        state = ProcessingState(
            session_id=session_id,
            started_at=datetime.now(),
            last_updated=datetime.now(),
            total_tasks=len(tasks),
            completed_tasks=0,
            failed_tasks=0,
            task_states=task_states,
            chunk_states={},  # 청크 상태는 나중에 추가
            configuration=configuration,
            progress_tracker=None  # 나중에 직렬화하여 저장
        )
        
        self.save_state(state)
        print(f"📝 새 세션 생성: {session_id}")
        return session_id
    
    def save_state(self, state: ProcessingState) -> bool:
        """상태를 파일에 저장"""
        try:
            # 기존 파일을 백업으로 이동
            if self.state_file.exists():
                self.state_file.replace(self.backup_file)
            
            # 새 상태 저장
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ 체크포인트 저장 실패: {e}")
            return False
    
    def load_state(self) -> Optional[ProcessingState]:
        """저장된 상태 로드"""
        try:
            if not self.state_file.exists():
                return None
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = ProcessingState.from_dict(data)
            self.current_session_id = state.session_id
            return state
            
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
            
            # 백업 파일로 복구 시도
            try:
                if self.backup_file.exists():
                    print("🔄 백업 파일에서 복구 시도 중...")
                    with open(self.backup_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    state = ProcessingState.from_dict(data)
                    self.current_session_id = state.session_id
                    print("✅ 백업에서 복구 성공")
                    return state
                    
            except Exception as backup_error:
                print(f"❌ 백업 복구도 실패: {backup_error}")
            
            return None
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          error_message: Optional[str] = None) -> bool:
        """작업 상태 업데이트"""
        state = self.load_state()
        if not state:
            print(f"❌ 상태를 로드할 수 없어 {task_id} 업데이트 실패")
            return False
        
        if task_id not in state.task_states:
            print(f"❌ 알 수 없는 작업 ID: {task_id}")
            return False
        
        # 상태 업데이트
        task_state = state.task_states[task_id]
        old_status = TaskStatus(task_state['status'])
        task_state['status'] = status.value
        task_state['error_message'] = error_message
        
        now = datetime.now()
        
        if status == TaskStatus.IN_PROGRESS and old_status == TaskStatus.NOT_STARTED:
            task_state['started_at'] = now.isoformat()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task_state['completed_at'] = now.isoformat()
            
            # 전체 통계 업데이트
            if old_status != status:  # 상태가 실제로 변경된 경우만
                if status == TaskStatus.COMPLETED:
                    state.completed_tasks += 1
                elif status == TaskStatus.FAILED:
                    state.failed_tasks += 1
        
        state.last_updated = now
        
        success = self.save_state(state)
        if success:
            print(f"📝 {task_id}: {old_status.value} → {status.value}")
        
        return success
    
    def get_resumable_tasks(self) -> List[str]:
        """재시작 가능한 작업 목록 반환"""
        state = self.load_state()
        if not state:
            return []
        
        resumable = []
        for task_id, task_state in state.task_states.items():
            status = TaskStatus(task_state['status'])
            if status in [TaskStatus.NOT_STARTED, TaskStatus.FAILED]:
                resumable.append(task_id)
        
        return resumable
    
    def get_completed_tasks(self) -> List[str]:
        """완료된 작업 목록 반환"""
        state = self.load_state()
        if not state:
            return []
        
        completed = []
        for task_id, task_state in state.task_states.items():
            status = TaskStatus(task_state['status'])
            if status == TaskStatus.COMPLETED:
                completed.append(task_id)
        
        return completed
    
    def clear_checkpoint(self) -> bool:
        """체크포인트 파일 삭제"""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            if self.backup_file.exists():
                self.backup_file.unlink()
            
            self.current_session_id = None
            print("🗑️ 체크포인트 파일 삭제됨")
            return True
            
        except Exception as e:
            print(f"❌ 체크포인트 삭제 실패: {e}")
            return False
    
    def get_progress_summary(self) -> Optional[Dict[str, Any]]:
        """진행 상황 요약 반환"""
        state = self.load_state()
        if not state:
            return None
        
        # 상태별 작업 수 계산
        status_counts = {}
        for task_state in state.task_states.values():
            status = task_state['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 진행률 계산
        progress_percent = (state.completed_tasks / state.total_tasks * 100) if state.total_tasks > 0 else 0
        
        # 경과 시간 계산
        elapsed_time = state.last_updated - state.started_at
        
        return {
            'session_id': state.session_id,
            'total_tasks': state.total_tasks,
            'completed_tasks': state.completed_tasks,
            'failed_tasks': state.failed_tasks,
            'progress_percent': progress_percent,
            'elapsed_time': str(elapsed_time),
            'status_counts': status_counts,
            'started_at': state.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'last_updated': state.last_updated.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def is_session_resumable(self) -> bool:
        """현재 세션이 재시작 가능한지 확인"""
        state = self.load_state()
        if not state:
            return False
        
        # 완료되지 않은 작업이 있는지 확인
        resumable_tasks = self.get_resumable_tasks()
        return len(resumable_tasks) > 0
    
    def create_chunk_states(self, pdf_name: str, total_pages: int, chunk_size: int) -> List[str]:
        """PDF용 청크 상태들을 생성"""
        state = self.load_state()
        if not state:
            return []
        
        total_chunks = (total_pages + chunk_size - 1) // chunk_size
        chunk_ids = []
        
        for chunk_index in range(total_chunks):
            page_start = chunk_index * chunk_size + 1
            page_end = min((chunk_index + 1) * chunk_size, total_pages)
            
            chunk_id = f"{pdf_name}_chunk_{chunk_index:03d}"
            
            chunk_state = ChunkState(
                chunk_id=chunk_id,
                pdf_name=pdf_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                page_start=page_start,
                page_end=page_end,
                status=TaskStatus.NOT_STARTED
            )
            
            state.chunk_states[chunk_id] = chunk_state
            chunk_ids.append(chunk_id)
        
        self.save_state(state)
        print(f"📦 {pdf_name}: {total_chunks}개 청크 생성 (페이지당 {chunk_size})")
        
        return chunk_ids
    
    def update_chunk_status(self, chunk_id: str, status: TaskStatus, 
                           processed_pages: List[int] = None,
                           error_message: Optional[str] = None) -> bool:
        """청크 상태 업데이트"""
        state = self.load_state()
        if not state or chunk_id not in state.chunk_states:
            print(f"❌ 알 수 없는 청크 ID: {chunk_id}")
            return False
        
        chunk_state = state.chunk_states[chunk_id]
        old_status = chunk_state.status
        chunk_state.status = status
        chunk_state.error_message = error_message
        
        now = datetime.now()
        
        if status == TaskStatus.IN_PROGRESS and old_status == TaskStatus.NOT_STARTED:
            chunk_state.started_at = now
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            chunk_state.completed_at = now
        
        if processed_pages:
            chunk_state.processed_pages = processed_pages
        
        state.last_updated = now
        
        success = self.save_state(state)
        if success:
            print(f"📝 {chunk_id}: {old_status.value} → {status.value}")
            
            # 진행 상황 추적기 업데이트
            if self.progress_tracker:
                if status == TaskStatus.COMPLETED:
                    self.progress_tracker.complete_task(chunk_id, success=True)
                elif status == TaskStatus.FAILED:
                    self.progress_tracker.complete_task(chunk_id, success=False)
        
        return success
    
    def get_resumable_chunks(self, pdf_name: Optional[str] = None) -> List[str]:
        """재시작 가능한 청크 목록 반환"""
        state = self.load_state()
        if not state:
            return []
        
        resumable = []
        for chunk_id, chunk_state in state.chunk_states.items():
            if pdf_name and chunk_state.pdf_name != pdf_name:
                continue
                
            if chunk_state.status in [TaskStatus.NOT_STARTED, TaskStatus.FAILED]:
                resumable.append(chunk_id)
        
        return resumable
    
    def get_chunk_progress_summary(self, pdf_name: Optional[str] = None) -> Dict[str, Any]:
        """청크 처리 진행 상황 요약"""
        state = self.load_state()
        if not state:
            return {}
        
        chunks = [chunk for chunk in state.chunk_states.values() 
                 if not pdf_name or chunk.pdf_name == pdf_name]
        
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        completed_chunks = sum(1 for chunk in chunks if chunk.status == TaskStatus.COMPLETED)
        failed_chunks = sum(1 for chunk in chunks if chunk.status == TaskStatus.FAILED)
        in_progress_chunks = sum(1 for chunk in chunks if chunk.status == TaskStatus.IN_PROGRESS)
        
        # 전체 페이지 진행 상황
        total_pages = sum(chunk.page_end - chunk.page_start + 1 for chunk in chunks)
        processed_pages = sum(len(chunk.processed_pages) for chunk in chunks)
        
        return {
            'pdf_name': pdf_name or 'ALL',
            'total_chunks': total_chunks,
            'completed_chunks': completed_chunks,
            'failed_chunks': failed_chunks,
            'in_progress_chunks': in_progress_chunks,
            'chunk_progress_percent': (completed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
            'total_pages': total_pages,
            'processed_pages': processed_pages,
            'page_progress_percent': (processed_pages / total_pages * 100) if total_pages > 0 else 0
        }
    
    def save_progress_checkpoint(self) -> bool:
        """진행 상황 체크포인트 저장 (중단 복구용)"""
        if not self.progress_tracker:
            return False
        
        state = self.load_state()
        if not state:
            return False
        
        # ProgressTracker 데이터를 직렬화하여 저장
        try:
            eta_summary = self.progress_tracker.get_eta_summary()
            performance_data = self.progress_tracker.performance_metrics.to_dict() if self.progress_tracker.performance_metrics else {}
            
            state.progress_tracker = {
                'session_id': self.progress_tracker.session_id,
                'eta_summary': eta_summary,
                'performance_metrics': performance_data
            }
            
            return self.save_state(state)
            
        except Exception as e:
            print(f"⚠️ 진행 상황 체크포인트 저장 실패: {e}")
            return False


if __name__ == "__main__":
    # 테스트 코드
    manager = CheckpointManager()
    
    # 테스트용 작업 생성
    from modules.models.task_models import PDFTask
    test_tasks = [
        PDFTask(
            pdf_path=Path("test1.pdf"),
            output_path=Path("test1.md")
        ),
        PDFTask(
            pdf_path=Path("test2.pdf"),
            output_path=Path("test2.md")
        )
    ]
    
    # 새 세션 생성
    session_id = manager.create_new_session(test_tasks)
    
    # 상태 업데이트 테스트
    manager.update_task_status("test1", TaskStatus.IN_PROGRESS)
    manager.update_task_status("test1", TaskStatus.COMPLETED)
    manager.update_task_status("test2", TaskStatus.FAILED, "테스트 오류")
    
    # 진행 상황 확인
    summary = manager.get_progress_summary()
    if summary:
        print("📊 진행 상황:")
        print(f"   전체: {summary['total_tasks']}")
        print(f"   완료: {summary['completed_tasks']}")
        print(f"   실패: {summary['failed_tasks']}")
        print(f"   진행률: {summary['progress_percent']:.1f}%")
    
    # 정리
    manager.clear_checkpoint()