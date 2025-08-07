#!/usr/bin/env python3
"""
성능 최적화된 설정으로 common.pdf 변환
"""
import config

def apply_performance_optimizations():
    """성능 최적화 설정 적용"""
    print("🔧 성능 최적화 설정 적용 중...")
    
    # 원본 설정 백업
    original_settings = {
        'DPI': config.DPI,
        'IMAGE_FORMAT': config.IMAGE_FORMAT,
        'SEMANTIC_CHUNKING': config.SEMANTIC_CHUNKING,
        'EXTRACT_CODE_SNIPPETS': config.EXTRACT_CODE_SNIPPETS
    }
    
    # 최적화 설정 적용
    config.DPI = 150
    config.IMAGE_FORMAT = "JPEG"
    config.SEMANTIC_CHUNKING = False  # 후처리 시간 단축
    config.EXTRACT_CODE_SNIPPETS = False  # 코드 추출 생략
    
    print(f"   DPI: {original_settings['DPI']} → {config.DPI}")
    print(f"   IMAGE_FORMAT: {original_settings['IMAGE_FORMAT']} → {config.IMAGE_FORMAT}")
    print(f"   SEMANTIC_CHUNKING: {original_settings['SEMANTIC_CHUNKING']} → {config.SEMANTIC_CHUNKING}")
    print(f"   EXTRACT_CODE_SNIPPETS: {original_settings['EXTRACT_CODE_SNIPPETS']} → {config.EXTRACT_CODE_SNIPPETS}")
    
    return original_settings

def restore_original_settings(original_settings):
    """원본 설정 복원"""
    print("\n🔄 원본 설정 복원 중...")
    
    config.DPI = original_settings['DPI']
    config.IMAGE_FORMAT = original_settings['IMAGE_FORMAT']
    config.SEMANTIC_CHUNKING = original_settings['SEMANTIC_CHUNKING']
    config.EXTRACT_CODE_SNIPPETS = original_settings['EXTRACT_CODE_SNIPPETS']
    
    print("   ✅ 원본 설정 복원 완료")

if __name__ == "__main__":
    # 현재 설정 확인
    print("📋 현재 설정:")
    print(f"   DPI: {config.DPI}")
    print(f"   IMAGE_FORMAT: {config.IMAGE_FORMAT}")
    print(f"   SEMANTIC_CHUNKING: {config.SEMANTIC_CHUNKING}")
    print(f"   EXTRACT_CODE_SNIPPETS: {config.EXTRACT_CODE_SNIPPETS}")
    
    # 최적화 설정 적용
    original = apply_performance_optimizations()
    
    print("\n✅ 최적화 설정 적용 완료")
    print("이제 다음 명령어로 변환을 실행하세요:")
    print("python main.py common")