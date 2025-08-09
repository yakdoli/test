"""
GPU 전용 워커 모듈 - 별도 프로세스에서 실행
"""

import torch
import os
import pickle
from pathlib import Path
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config


def process_image_batch_worker(data):
    """워커 프로세스에서 실행될 함수"""
    device_id, image_batch_serialized = data
    
    try:
        # CUDA 디바이스 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        torch.cuda.set_device(0)
        
        # 이미지 배치 역직렬화
        image_paths = pickle.loads(image_batch_serialized)
        
        print(f"🔧 GPU {device_id} 워커 초기화 중...")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 프로세서 로드
        processor = AutoProcessor.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 모델 로드 (현재 프로세스의 GPU만 사용)
        load_kwargs = {
            "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
            "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
            "device_map": "cuda:0",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True
        }
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        model.eval()
        
        print(f"✅ GPU {device_id} 워커 초기화 완료")
        
        # 배치 처리
        results = []
        for image_path in image_paths:
            try:
                # 프롬프트 생성
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": get_optimized_prompt()}
                    ]
                }]
                
                # 전처리
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                
                # GPU로 이동
                inputs = {k: v.to("cuda:0") if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
                
                # 입력 검증
                if 'input_ids' not in inputs:
                    raise ValueError("input_ids not found in inputs")
                
                # 생성 설정
                generation_config = {
                    "max_new_tokens": 2500,
                    "do_sample": False,
                    "temperature": 0.1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                
                # 텍스트 생성
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generation_config)
                
                # 응답 추출
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                results.append(output_text)
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ GPU {device_id} 이미지 처리 실패 ({image_path}): {e}")
                results.append(f"<!-- GPU {device_id} 처리 실패: {image_path.name} -->")
        
        print(f"✅ GPU {device_id} 배치 처리 완료 ({len(results)}개)")
        return results
        
    except Exception as e:
        print(f"❌ GPU {device_id} 워커 오류: {e}")
        return [f"<!-- GPU {device_id} 워커 실패 -->"]


def get_optimized_prompt() -> str:
    """최적화된 프롬프트"""
    return """Convert this technical documentation to markdown format.

Key requirements:
- Extract code snippets with proper language tags (```csharp, ```xml)
- Use clear heading structure (# ## ###)  
- Include all visible text and technical details
- Format tables and lists properly
- Add code examples with proper formatting

Focus on accuracy and completeness."""