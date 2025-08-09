"""
Qwen2.5-VL 모델 로딩 수정
올바른 클래스를 사용합니다
"""

import torch
import traceback

def test_correct_model_class():
    """올바른 모델 클래스 찾기"""
    print("🔍 올바른 모델 클래스 확인")
    
    # 가능한 클래스들 시도
    classes_to_try = [
        ("Qwen2_5_VLForConditionalGeneration", "transformers"),
        ("Qwen2_5VLForConditionalGeneration", "transformers"),
        ("Qwen25VLForConditionalGeneration", "transformers"),
        ("AutoModelForVision2Seq", "transformers"),
        ("AutoModelForCausalLM", "transformers")
    ]
    
    for class_name, module_name in classes_to_try:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            print(f"✅ {class_name} 클래스 사용 가능")
            return model_class, class_name
        except AttributeError:
            print(f"❌ {class_name} 클래스 없음")
        except Exception as e:
            print(f"❌ {class_name} 오류: {e}")
    
    return None, None

def test_auto_model():
    """AutoModel을 사용한 로딩"""
    print("\n🔍 AutoModel을 사용한 로딩 테스트")
    try:
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        print("✅ AutoModel 로드 성공")
        print(f"   모델 타입: {type(model)}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ AutoModel 로드 실패: {e}")
        traceback.print_exc()
        return False

def test_alternative_loading():
    """대안적 로딩 방법"""
    print("\n🔍 대안적 로딩 방법 테스트")
    try:
        from transformers import AutoModelForCausalLM
        
        # trust_remote_code=True로 자동 클래스 매핑
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        print("✅ AutoModelForCausalLM 로드 성공")
        print(f"   실제 모델 타입: {type(model)}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ AutoModelForCausalLM 로드 실패: {e}")
        traceback.print_exc()
        return False

def test_with_revision():
    """특정 리비전 사용"""
    print("\n🔍 특정 리비전 사용 테스트")
    try:
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32,
            revision="main"
        )
        
        print("✅ 특정 리비전 로드 성공")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 특정 리비전 로드 실패: {e}")
        traceback.print_exc()
        return False

def test_compatible_model():
    """호환되는 모델 테스트"""
    print("\n🔍 호환되는 모델 테스트")
    
    # 대안 모델들
    alternative_models = [
        "Qwen/Qwen2-VL-7B-Instruct",  # 2.0 버전
        "Qwen/Qwen-VL-Chat",           # 원래 버전
    ]
    
    for model_name in alternative_models:
        print(f"\n🔄 {model_name} 테스트:")
        try:
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            print(f"✅ {model_name} 로드 성공")
            print(f"   모델 타입: {type(model)}")
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()
            
            return True, model_name
            
        except Exception as e:
            print(f"❌ {model_name} 로드 실패: {e}")
    
    return False, None

def main():
    print("🚀 Qwen2.5-VL 모델 로딩 수정 테스트")
    print("=" * 50)
    
    # 1. 올바른 클래스 찾기
    model_class, class_name = test_correct_model_class()
    
    # 2. AutoModel 시도
    auto_success = test_auto_model()
    
    if not auto_success:
        # 3. AutoModelForCausalLM 시도
        causal_success = test_alternative_loading()
        
        if not causal_success:
            # 4. 특정 리비전 시도
            revision_success = test_with_revision()
            
            if not revision_success:
                # 5. 호환되는 모델 시도
                compatible_success, compatible_model = test_compatible_model()
                
                if compatible_success:
                    print(f"\n🎉 호환되는 모델 발견: {compatible_model}")
                else:
                    print(f"\n❌ 모든 시도 실패")
            else:
                print(f"\n🎉 특정 리비전으로 로드 성공!")
        else:
            print(f"\n🎉 AutoModelForCausalLM으로 로드 성공!")
    else:
        print(f"\n🎉 AutoModel로 로드 성공!")

if __name__ == "__main__":
    main()