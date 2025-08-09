"""
Qwen2.5-VL 모델 로딩 디버깅
단계별로 문제점을 확인합니다
"""

import torch
import traceback
from pathlib import Path

def test_step1_tokenizer():
    """1단계: 토크나이저 로딩"""
    print("🔍 1단계: 토크나이저 로딩 테스트")
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        print("✅ 토크나이저 로드 성공")
        print(f"   어휘 크기: {len(tokenizer)}")
        return tokenizer
        
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        traceback.print_exc()
        return None

def test_step2_processor():
    """2단계: 프로세서 로딩"""
    print("\n🔍 2단계: 프로세서 로딩 테스트")
    try:
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        print("✅ 프로세서 로드 성공")
        return processor
        
    except Exception as e:
        print(f"❌ 프로세서 로드 실패: {e}")
        traceback.print_exc()
        return None

def test_step3_config():
    """3단계: 모델 설정 확인"""
    print("\n🔍 3단계: 모델 설정 확인")
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )
        
        print("✅ 모델 설정 로드 성공")
        print(f"   모델 타입: {config.model_type}")
        print(f"   숨겨진 크기: {config.hidden_size}")
        print(f"   어텐션 헤드: {config.num_attention_heads}")
        print(f"   레이어 수: {config.num_hidden_layers}")
        
        return config
        
    except Exception as e:
        print(f"❌ 모델 설정 로드 실패: {e}")
        traceback.print_exc()
        return None

def test_step4_model_class():
    """4단계: 모델 클래스 확인"""
    print("\n🔍 4단계: 모델 클래스 확인")
    try:
        # 새로운 방식으로 시도
        from transformers import Qwen2VLForConditionalGeneration
        print("✅ Qwen2VLForConditionalGeneration 클래스 사용 가능")
        return Qwen2VLForConditionalGeneration
        
    except ImportError:
        try:
            # 구 버전 방식
            from transformers import AutoModelForCausalLM
            print("⚠️ AutoModelForCausalLM로 대체 시도")
            return AutoModelForCausalLM
        except Exception as e:
            print(f"❌ 모델 클래스 로드 실패: {e}")
            return None

def test_step5_model_loading():
    """5단계: 실제 모델 로딩"""
    print("\n🔍 5단계: 모델 로딩 테스트 (CPU 전용)")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        
        # CPU에서 먼저 시도
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        print("✅ 모델 CPU 로드 성공")
        print(f"   모델 매개변수: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 CPU 로드 실패: {e}")
        traceback.print_exc()
        return False

def test_step6_gpu_loading():
    """6단계: GPU 로딩 테스트"""
    print("\n🔍 6단계: GPU 모델 로딩 테스트")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        
        print("GPU 메모리 정리 중...")
        torch.cuda.empty_cache()
        
        # GPU에서 시도 (단일 GPU)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.float16
        )
        
        print("✅ 모델 GPU 로드 성공")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 GPU 로드 실패: {e}")
        traceback.print_exc()
        return False

def test_step7_auto_device():
    """7단계: Auto device 테스트"""
    print("\n🔍 7단계: Auto device 맵핑 테스트")
    try:
        from transformers import Qwen2VLForConditionalGeneration
        
        print("GPU 메모리 정리 중...")
        torch.cuda.empty_cache()
        
        # Auto device mapping
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✅ 모델 Auto device 로드 성공")
        print(f"   디바이스 맵: {model.hf_device_map}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 Auto device 로드 실패: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 Qwen2.5-VL 모델 로딩 디버깅 시작")
    print("=" * 50)
    
    # 단계별 테스트
    tokenizer = test_step1_tokenizer()
    processor = test_step2_processor()
    config = test_step3_config()
    model_class = test_step4_model_class()
    
    if tokenizer and processor and config and model_class:
        print("\n✅ 기본 구성요소 모두 로드 성공!")
        
        # 실제 모델 로딩 테스트
        cpu_success = test_step5_model_loading()
        if cpu_success:
            gpu_success = test_step6_gpu_loading()
            if gpu_success:
                auto_success = test_step7_auto_device()
                
                if auto_success:
                    print("\n🎉 모든 테스트 성공! 모델 로딩 가능합니다.")
                else:
                    print("\n⚠️ Auto device 맵핑에 문제가 있습니다.")
            else:
                print("\n⚠️ GPU 로딩에 문제가 있습니다.")
        else:
            print("\n❌ CPU 로딩도 실패했습니다.")
    else:
        print("\n❌ 기본 구성요소 로딩에 문제가 있습니다.")
    
    # 메모리 상태 확인
    if torch.cuda.is_available():
        print(f"\n💾 GPU 메모리 상태:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.1f}GB 할당됨, {cached:.1f}GB 캐시됨, {total:.1f}GB 총 메모리")

if __name__ == "__main__":
    main()