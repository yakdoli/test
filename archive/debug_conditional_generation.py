"""
조건부 생성 모델 클래스 확인
"""

import torch

def test_conditional_generation_models():
    """조건부 생성 모델 클래스들 테스트"""
    print("🔍 조건부 생성 모델 클래스 확인")
    
    # 가능한 조건부 생성 클래스들
    classes_to_try = [
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForVision2Seq", 
        "AutoModelForSeq2SeqLM",
        "AutoModelForCausalLM"
    ]
    
    successful_classes = []
    
    for class_name in classes_to_try:
        print(f"\n🔄 {class_name} 테스트:")
        try:
            from transformers import AutoConfig
            import transformers
            
            # 클래스 가져오기
            model_class = getattr(transformers, class_name)
            
            # 설정으로 호환성 확인
            config = AutoConfig.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                trust_remote_code=True
            )
            
            print(f"   설정 호환성: ✅")
            print(f"   아키텍처: {config.architectures}")
            
            # 실제 모델 로드 시도 (CPU, 작은 메모리)
            model = model_class.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
            print(f"   모델 로드: ✅")
            print(f"   실제 타입: {type(model)}")
            
            # generate 메서드 확인
            if hasattr(model, 'generate'):
                print(f"   generate 메서드: ✅")
                successful_classes.append((class_name, model_class))
            else:
                print(f"   generate 메서드: ❌")
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   오류: {e}")
    
    return successful_classes

def test_direct_qwen25_class():
    """직접 Qwen2_5_VL 클래스 테스트"""
    print("\n🔍 직접 Qwen2_5_VLForConditionalGeneration 클래스 테스트")
    
    try:
        # 직접 import
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        print("✅ 직접 Qwen2_5_VLForConditionalGeneration 로드 성공")
        print(f"   모델 타입: {type(model)}")
        print(f"   generate 메서드: {'✅' if hasattr(model, 'generate') else '❌'}")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 직접 클래스 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_loading():
    """GPU 로딩 테스트"""
    print("\n🔍 GPU 로딩 테스트")
    
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.float16
        )
        
        print("✅ GPU 로딩 성공")
        print(f"   모델 디바이스: {next(model.parameters()).device}")
        
        # 메모리 상태 확인
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   GPU 메모리 사용: {allocated:.1f}GB")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 조건부 생성 모델 클래스 확인")
    print("=" * 50)
    
    # 1. 조건부 생성 모델 클래스들 테스트
    successful_classes = test_conditional_generation_models()
    
    if successful_classes:
        print(f"\n✅ 성공한 클래스들:")
        for class_name, model_class in successful_classes:
            print(f"   - {class_name}: {model_class}")
    
    # 2. 직접 Qwen2.5 클래스 테스트
    direct_success = test_direct_qwen25_class()
    
    # 3. GPU 로딩 테스트
    if direct_success:
        gpu_success = test_gpu_loading()
        
        if gpu_success:
            print(f"\n🎉 모든 테스트 성공! GPU 로딩 가능합니다.")
        else:
            print(f"\n⚠️ CPU는 가능하지만 GPU 로딩에 문제가 있습니다.")
    else:
        print(f"\n❌ 직접 클래스 로딩에 문제가 있습니다.")

if __name__ == "__main__":
    main()