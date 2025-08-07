#!/usr/bin/env python3
"""
마크다운 변환 테스트 결과 분석 및 실행 계획
"""
import config
from pathlib import Path

def analyze_test_results():
    """테스트 결과 분석"""
    print("📊 qwen2.5vl 마크다운 변환 테스트 분석")
    print("=" * 80)
    
    # 테스트 결과 데이터
    test_results = {
        'DICOM': {'avg_time': 7.1, 'success_rate': 100, 'pages': 12},
        'Gauge': {'avg_time': 5.0, 'success_rate': 100, 'pages': 31},
        'PDF Viewer': {'avg_time': 12.8, 'success_rate': 100, 'pages': 29},
        'common': {'avg_time': 8.0, 'success_rate': 100, 'pages': 145}
    }
    
    print("🎯 테스트 결과 요약:")
    print(f"   • 테스트된 문서: 4개")
    print(f"   • 테스트된 이미지: 8개")
    print(f"   • 전체 성공률: 100%")
    print(f"   • 평균 변환 시간: 8.2초/페이지")
    
    print(f"\n📈 성능 개선 확인:")
    print(f"   • 이전 common.pdf 테스트: 137.3초/페이지")
    print(f"   • 현재 common 테스트: 8.0초/페이지")
    print(f"   • 성능 향상: {137.3/8.0:.1f}배 빠름! 🚀")
    
    print(f"\n📋 문서별 예상 변환 시간 (업데이트):")
    for doc_name, data in test_results.items():
        estimated_time = data['pages'] * data['avg_time'] / 60  # 분
        print(f"   • {doc_name:<15} {data['pages']:>3}p × {data['avg_time']:>4.1f}초 = {estimated_time:>5.1f}분")
    
    return test_results

def calculate_updated_estimates():
    """업데이트된 전체 변환 시간 예상"""
    print(f"\n⏱️ 전체 변환 시간 재계산")
    print("-" * 50)
    
    # 스테이징된 모든 문서 정보
    staging_dir = config.STAGING_DIR
    
    # 테스트 결과 기반 평균 시간 (8.2초/페이지)
    avg_time_per_page = 8.2
    
    documents = []
    total_pages = 0
    total_time = 0
    
    for staged_dir in staging_dir.iterdir():
        if staged_dir.is_dir() and staged_dir.name != 'common_test':
            doc_name = staged_dir.name
            image_count = len(list(staged_dir.glob(f"*.{config.IMAGE_FORMAT.lower()}")))
            
            estimated_minutes = image_count * avg_time_per_page / 60
            
            documents.append({
                'name': doc_name,
                'pages': image_count,
                'minutes': estimated_minutes,
                'hours': estimated_minutes / 60
            })
            
            total_pages += image_count
            total_time += estimated_minutes
    
    # 크기별 정렬
    documents.sort(key=lambda x: x['pages'])
    
    print(f"📊 업데이트된 변환 시간 예상:")
    print(f"   총 페이지: {total_pages:,}개")
    print(f"   총 예상 시간: {total_time:.1f}분 ({total_time/60:.1f}시간)")
    print(f"   이전 예상: 285.9시간 → 현재 예상: {total_time/60:.1f}시간")
    print(f"   시간 단축: {285.9/(total_time/60):.1f}배! 🎉")
    
    # 카테고리별 분류
    quick_docs = [d for d in documents if d['hours'] < 1]
    short_docs = [d for d in documents if 1 <= d['hours'] < 5]
    medium_docs = [d for d in documents if 5 <= d['hours'] < 15]
    long_docs = [d for d in documents if d['hours'] >= 15]
    
    print(f"\n📦 업데이트된 배치 분류:")
    print(f"   Quick (<1시간):   {len(quick_docs):>2}개 문서")
    print(f"   Short (1-5시간):  {len(short_docs):>2}개 문서")
    print(f"   Medium (5-15시간): {len(medium_docs):>2}개 문서")
    print(f"   Long (15시간+):   {len(long_docs):>2}개 문서")
    
    return documents

def create_optimized_execution_plan(documents):
    """최적화된 실행 계획 생성"""
    print(f"\n🚀 최적화된 실행 계획")
    print("=" * 80)
    
    # 크기별 분류
    quick_docs = [d for d in documents if d['hours'] < 1]
    short_docs = [d for d in documents if 1 <= d['hours'] < 5]
    medium_docs = [d for d in documents if 5 <= d['hours'] < 15]
    long_docs = [d for d in documents if d['hours'] >= 15]
    
    print(f"🎯 Phase 1: Quick Wins (즉시 실행 - {len(quick_docs)}개)")
    for doc in quick_docs[:5]:  # 상위 5개만 표시
        print(f"   python main.py \"{doc['name']}\"  # {doc['minutes']:.1f}분")
    
    print(f"\n⚡ Phase 2: Short Term (오늘 완료 가능 - {len(short_docs)}개)")
    for doc in short_docs[:3]:  # 상위 3개만 표시
        print(f"   nohup python main.py \"{doc['name']}\" > {doc['name'].replace(' ', '_')}.log 2>&1 &")
    
    print(f"\n🔄 Phase 3: Medium Term (1-2일 소요 - {len(medium_docs)}개)")
    for doc in medium_docs[:2]:  # 상위 2개만 표시
        print(f"   nohup python main.py \"{doc['name']}\" > {doc['name'].replace(' ', '_')}.log 2>&1 &")
    
    print(f"\n⏰ Phase 4: Long Term (주말 처리 - {len(long_docs)}개)")
    for doc in long_docs:
        print(f"   # {doc['name']}: {doc['hours']:.1f}시간 예상")
    
    # 즉시 실행 가능한 명령어
    print(f"\n💡 즉시 실행 가능한 명령어:")
    if quick_docs:
        first_doc = quick_docs[0]
        print(f"   python main.py \"{first_doc['name']}\"")
        print(f"   # 예상 시간: {first_doc['minutes']:.1f}분")

def create_monitoring_commands():
    """모니터링 명령어 생성"""
    print(f"\n📊 모니터링 명령어")
    print("-" * 40)
    
    commands = [
        "# 실행 중인 변환 확인",
        "ps aux | grep 'python main.py'",
        "",
        "# 완료된 문서 확인", 
        "ls -la output/*.md",
        "",
        "# 로그 실시간 확인",
        "tail -f *.log",
        "",
        "# 디스크 사용량 확인",
        "df -h .",
        "",
        "# 변환 진행률 확인",
        "find output -name '*.md' | wc -l"
    ]
    
    for cmd in commands:
        print(f"   {cmd}")

def main():
    """메인 함수"""
    # 1. 테스트 결과 분석
    test_results = analyze_test_results()
    
    # 2. 업데이트된 시간 계산
    documents = calculate_updated_estimates()
    
    # 3. 최적화된 실행 계획
    create_optimized_execution_plan(documents)
    
    # 4. 모니터링 명령어
    create_monitoring_commands()
    
    # 5. 최종 권장사항
    print(f"\n🎉 최종 권장사항")
    print("=" * 80)
    print(f"✅ qwen2.5vl 성능이 예상보다 훨씬 우수함!")
    print(f"   • 이전 예상: 285.9시간 → 현재 예상: ~17시간")
    print(f"   • 성능 향상: 약 17배!")
    
    print(f"\n🚀 즉시 시작 권장:")
    if documents:
        smallest_doc = min(documents, key=lambda x: x['pages'])
        print(f"   python main.py \"{smallest_doc['name']}\"")
        print(f"   # {smallest_doc['pages']}페이지, {smallest_doc['minutes']:.1f}분 예상")
    
    print(f"\n📈 효율적인 진행 방법:")
    print(f"   1. Quick Wins부터 시작하여 시스템 안정성 확인")
    print(f"   2. 성공 확인 후 Short/Medium Term 병렬 실행")
    print(f"   3. Long Term은 백그라운드에서 순차 실행")

if __name__ == "__main__":
    main()