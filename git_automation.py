"""
Git 자동화 유틸리티
태스크별 커밋 및 푸시 기능 제공
"""

import subprocess
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import config


class GitAutomation:
    """Git 작업 자동화 클래스"""
    
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or config.BASE_DIR
        self.commit_history = []
        self.commit_log_file = self.repo_path / ".git_automation_log.json"
        self._load_commit_history()
    
    def _load_commit_history(self):
        """커밋 기록 로드"""
        if self.commit_log_file.exists():
            try:
                with open(self.commit_log_file, 'r', encoding='utf-8') as f:
                    self.commit_history = json.load(f)
            except Exception as e:
                print(f"⚠️ 커밋 기록 로드 실패: {e}")
                self.commit_history = []
    
    def _save_commit_history(self):
        """커밋 기록 저장"""
        try:
            with open(self.commit_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.commit_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 커밋 기록 저장 실패: {e}")
    
    def _run_git_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Git 명령 실행"""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Git 명령 실행 실패: {' '.join(command)}")
            print(f"   오류 메시지: {e.stderr}")
            raise
    
    def get_git_status(self) -> Dict[str, List[str]]:
        """Git 상태 확인"""
        try:
            result = self._run_git_command(['status', '--porcelain'])
            
            status = {
                'modified': [],
                'added': [],
                'deleted': [],
                'untracked': [],
                'renamed': []
            }
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                
                status_code = line[:2]
                filename = line[3:].strip()
                
                if status_code.startswith('M'):
                    status['modified'].append(filename)
                elif status_code.startswith('A'):
                    status['added'].append(filename)
                elif status_code.startswith('D'):
                    status['deleted'].append(filename)
                elif status_code.startswith('??'):
                    status['untracked'].append(filename)
                elif status_code.startswith('R'):
                    status['renamed'].append(filename)
            
            return status
            
        except subprocess.CalledProcessError:
            print("❌ Git 상태 확인 실패")
            return {}
    
    def get_current_branch(self) -> str:
        """현재 브랜치 확인"""
        try:
            result = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def stage_files(self, files: List[str] = None) -> bool:
        """파일들을 스테이징"""
        try:
            if files:
                for file in files:
                    self._run_git_command(['add', file])
            else:
                # 모든 변경 사항 스테이징
                self._run_git_command(['add', '.'])
            
            print(f"✅ 파일 스테이징 완료: {len(files) if files else '전체'}")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ 파일 스테이징 실패")
            return False
    
    def create_task_commit(self, task_name: str, task_description: str, 
                          files: List[str] = None) -> bool:
        """태스크별 커밋 생성"""
        
        # 변경 사항 확인
        status = self.get_git_status()
        if not any(status.values()):
            print("📋 커밋할 변경 사항이 없습니다.")
            return False
        
        try:
            # 파일 스테이징
            if not self.stage_files(files):
                return False
            
            # 커밋 메시지 생성
            commit_message = self._generate_task_commit_message(
                task_name, task_description, status
            )
            
            # 커밋 실행
            self._run_git_command(['commit', '-m', commit_message])
            
            # 커밋 기록 저장
            commit_record = {
                'task_name': task_name,
                'task_description': task_description,
                'commit_message': commit_message,
                'timestamp': datetime.now().isoformat(),
                'branch': self.get_current_branch(),
                'files_changed': sum(len(file_list) for file_list in status.values())
            }
            
            self.commit_history.append(commit_record)
            self._save_commit_history()
            
            print(f"✅ 태스크 커밋 완료: {task_name}")
            print(f"   브랜치: {commit_record['branch']}")
            print(f"   변경 파일: {commit_record['files_changed']}개")
            
            return True
            
        except subprocess.CalledProcessError:
            print(f"❌ 태스크 커밋 실패: {task_name}")
            return False
    
    def _generate_task_commit_message(self, task_name: str, task_description: str,
                                    status: Dict[str, List[str]]) -> str:
        """태스크별 커밋 메시지 생성"""
        
        # 메인 메시지
        main_msg = f"feat: {task_name}"
        
        # 상세 설명 추가
        details = []
        details.append(f"{task_description}")
        details.append("")
        
        # 변경 사항 요약
        if status['added'] or status['untracked']:
            new_files = status['added'] + status['untracked']
            details.append(f"새 파일 추가: {len(new_files)}개")
            for file in new_files[:5]:  # 최대 5개만 표시
                details.append(f"  + {file}")
            if len(new_files) > 5:
                details.append(f"  ... 및 {len(new_files) - 5}개 더")
            details.append("")
        
        if status['modified']:
            details.append(f"수정된 파일: {len(status['modified'])}개")
            for file in status['modified'][:5]:  # 최대 5개만 표시
                details.append(f"  ~ {file}")
            if len(status['modified']) > 5:
                details.append(f"  ... 및 {len(status['modified']) - 5}개 더")
            details.append("")
        
        if status['deleted']:
            details.append(f"삭제된 파일: {len(status['deleted'])}개")
            for file in status['deleted']:
                details.append(f"  - {file}")
            details.append("")
        
        # 자동 생성 표시
        details.append("🤖 PDF to Markdown 변환기 프로젝트 개선")
        details.append("")
        details.append("Co-authored-by: Claude <noreply@anthropic.com>")
        
        return main_msg + "\n\n" + "\n".join(details)
    
    def push_to_remote(self, remote: str = "origin", branch: str = None) -> bool:
        """원격 저장소에 푸시"""
        try:
            current_branch = branch or self.get_current_branch()
            
            # 원격 브랜치 존재 확인
            try:
                self._run_git_command(['ls-remote', '--exit-code', remote, current_branch])
                push_command = ['push', remote, current_branch]
            except subprocess.CalledProcessError:
                # 새 브랜치인 경우 upstream 설정
                push_command = ['push', '-u', remote, current_branch]
            
            self._run_git_command(push_command)
            
            print(f"✅ 원격 저장소 푸시 완료: {remote}/{current_branch}")
            return True
            
        except subprocess.CalledProcessError:
            print(f"❌ 원격 저장소 푸시 실패: {remote}/{current_branch}")
            return False
    
    def create_feature_branch(self, feature_name: str) -> bool:
        """기능 브랜치 생성"""
        try:
            branch_name = f"feature/{feature_name.lower().replace(' ', '-')}"
            
            # 브랜치 생성 및 체크아웃
            self._run_git_command(['checkout', '-b', branch_name])
            
            print(f"✅ 기능 브랜치 생성: {branch_name}")
            return True
            
        except subprocess.CalledProcessError:
            print(f"❌ 기능 브랜치 생성 실패: {feature_name}")
            return False
    
    def commit_and_push_task(self, task_name: str, task_description: str,
                           files: List[str] = None, create_branch: bool = False,
                           push_to_remote: bool = True) -> bool:
        """태스크 커밋 및 푸시 (원스톱)"""
        
        print(f"\n🚀 태스크 '{task_name}' Git 작업 시작")
        print("=" * 50)
        
        try:
            # 기능 브랜치 생성 (옵션)
            if create_branch:
                if not self.create_feature_branch(task_name):
                    return False
            
            # 커밋 생성
            if not self.create_task_commit(task_name, task_description, files):
                return False
            
            # 원격 푸시 (옵션)
            if push_to_remote:
                if not self.push_to_remote():
                    print("⚠️ 푸시는 실패했지만 커밋은 성공했습니다.")
                    return True  # 커밋은 성공했으므로 True 반환
            
            print(f"🎉 태스크 '{task_name}' Git 작업 완료")
            return True
            
        except Exception as e:
            print(f"❌ Git 작업 중 예상치 못한 오류: {e}")
            return False
    
    def get_commit_summary(self) -> Dict[str, Any]:
        """커밋 요약 정보 반환"""
        if not self.commit_history:
            return {}
        
        total_commits = len(self.commit_history)
        recent_commits = self.commit_history[-5:]  # 최근 5개
        
        return {
            'total_commits': total_commits,
            'current_branch': self.get_current_branch(),
            'recent_commits': [
                {
                    'task_name': commit['task_name'],
                    'timestamp': commit['timestamp'],
                    'files_changed': commit['files_changed']
                }
                for commit in recent_commits
            ]
        }
    
    def print_status(self):
        """현재 Git 상태 출력"""
        print("\n📊 Git 상태 요약")
        print("=" * 30)
        
        # 브랜치 정보
        current_branch = self.get_current_branch()
        print(f"현재 브랜치: {current_branch}")
        
        # 변경 사항 확인
        status = self.get_git_status()
        if any(status.values()):
            print("\n📝 변경 사항:")
            for status_type, files in status.items():
                if files:
                    print(f"  {status_type}: {len(files)}개")
        else:
            print("\n✅ 변경 사항 없음")
        
        # 커밋 기록 요약
        summary = self.get_commit_summary()
        if summary:
            print(f"\n📈 커밋 통계:")
            print(f"  총 태스크 커밋: {summary['total_commits']}개")
            
            if summary['recent_commits']:
                print(f"  최근 커밋:")
                for commit in summary['recent_commits']:
                    timestamp = datetime.fromisoformat(commit['timestamp']).strftime('%Y-%m-%d %H:%M')
                    print(f"    • {commit['task_name']} ({timestamp}) - {commit['files_changed']}개 파일")


def create_git_automation() -> GitAutomation:
    """Git 자동화 인스턴스 생성"""
    return GitAutomation()


if __name__ == "__main__":
    # 테스트 코드
    git_auto = GitAutomation()
    
    print("🧪 Git 자동화 테스트")
    git_auto.print_status()
    
    # 테스트 커밋 (실제로는 실행하지 않음)
    # git_auto.commit_and_push_task(
    #     "테스트 태스크",
    #     "Git 자동화 기능 테스트",
    #     create_branch=False,
    #     push_to_remote=False
    # )