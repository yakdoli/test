"""
Flash Attention 2 효과 테스트
"""

import torch
import time
import psutil
from pathlib import Path
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import config

class FlashAttentionTester:
    def __init__(self):
        self.device = "cuda:0"
        torch.cuda.set_device(0)
        
    def load_model_with_attention(self, use_flash_attention: bool):
        """지정된 attention 타입으로 모델 로드"""
        print(f"{'Flash Attention 2' if use_flash_attention else 'Standard Attention'} 모델 로드 중...")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        # 토크나이저와 프로세서 로드
        tokenizer = AutoTokenizer.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        processor = AutoProcessor.from_pretrained(
            config.QWEN_MODEL_PATH,
            trust_remote_code=config.QWEN_TRUST_REMOTE_CODE
        )
        
        # 모델 로드 설정
        load_kwargs = {
            "pretrained_model_name_or_path": config.QWEN_MODEL_PATH,
            "trust_remote_code": config.QWEN_TRUST_REMOTE_CODE,
            "device_map": self.device,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            load_kwargs["attn_implementation"] = "eager"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**load_kwargs)
        model.eval()
        
        return model, tokenizer, processor
    
    def get_memory_stats(self):
        """GPU 메모리 사용량 반환"""
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        return {"allocated": allocated, "cached": cached}
    
    def test_single_image(self, model, tokenizer, processor, image_path: Path, use_flash: bool):
        """단일 이미지 처리 테스트"""
        print(f"\\n{'=' * 50}")
        print(f"{'Flash Attention 2' if use_flash else 'Standard Attention'} 테스트")
        print(f"{'=' * 50}")
        
        # 메모리 사용량 (모델 로드 후)
        memory_before = self.get_memory_stats()
        print(f"모델 로드 후 GPU 메모리: {memory_before['allocated']:.2f}GB (캐시: {memory_before['cached']:.2f}GB)")
        
        # 프롬프트 준비
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Convert this image to markdown format with proper structure."}
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
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 전처리 후 메모리
        memory_after_preprocess = self.get_memory_stats()
        print(f"전처리 후 GPU 메모리: {memory_after_preprocess['allocated']:.2f}GB (캐시: {memory_after_preprocess['cached']:.2f}GB)")
        
        # 생성 설정
        generation_config = {
            "max_new_tokens": 1000,  # 테스트용으로 줄임
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        # 추론 시간 측정
        start_time = time.time()
        peak_memory = 0
        
        with torch.no_grad():
            # 메모리 모니터링하면서 추론 실행
            for _ in range(1):  # warmup
                generated_ids = model.generate(**inputs, **generation_config)
                current_memory = self.get_memory_stats()['allocated']
                peak_memory = max(peak_memory, current_memory)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 응답 생성
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 최종 메모리
        memory_final = self.get_memory_stats()
        
        # 결과 출력
        print(f"\\n📊 성능 결과:")
        print(f"  ⏱️ 추론 시간: {inference_time:.2f}초")
        print(f"  🔥 피크 메모리: {peak_memory:.2f}GB")
        print(f"  📝 생성된 토큰: {len(generated_ids_trimmed[0])} 토큰")
        print(f"  📄 출력 길이: {len(output_text)} 문자")
        print(f"  💾 최종 메모리: {memory_final['allocated']:.2f}GB (캐시: {memory_final['cached']:.2f}GB)")
        
        # 메모리 효율성
        memory_efficiency = (peak_memory - memory_before['allocated']) / memory_before['allocated'] * 100
        print(f"  📈 메모리 증가: {memory_efficiency:.1f}%")
        
        return {
            "inference_time": inference_time,
            "peak_memory": peak_memory,
            "memory_efficiency": memory_efficiency,
            "output_length": len(output_text),
            "tokens_generated": len(generated_ids_trimmed[0])
        }
    
    def run_comparison_test(self, image_path: Path):
        """Flash Attention vs Standard Attention 비교 테스트"""
        print("🚀 Flash Attention 효과 비교 테스트 시작")
        print("=" * 80)
        
        results = {}
        
        # 1. Standard Attention 테스트
        print("\\n1️⃣ Standard Attention 테스트 중...")
        model_std, tokenizer_std, processor_std = self.load_model_with_attention(False)
        results['standard'] = self.test_single_image(model_std, tokenizer_std, processor_std, image_path, False)
        
        # 메모리 정리
        del model_std, tokenizer_std, processor_std
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # 2. Flash Attention 테스트
        print("\\n2️⃣ Flash Attention 2 테스트 중...")
        model_flash, tokenizer_flash, processor_flash = self.load_model_with_attention(True)
        results['flash'] = self.test_single_image(model_flash, tokenizer_flash, processor_flash, image_path, True)
        
        # 메모리 정리
        del model_flash, tokenizer_flash, processor_flash
        torch.cuda.empty_cache()
        
        # 3. 비교 결과
        self.print_comparison_results(results)
        
        return results
    
    def print_comparison_results(self, results):
        """비교 결과 출력"""
        print("\\n" + "=" * 80)
        print("🏆 Flash Attention vs Standard Attention 비교 결과")
        print("=" * 80)
        
        std = results['standard']
        flash = results['flash']
        
        # 속도 비교
        speed_improvement = (std['inference_time'] - flash['inference_time']) / std['inference_time'] * 100
        print(f"\\n⚡ 추론 속도:")
        print(f"  Standard:  {std['inference_time']:.2f}초")
        print(f"  Flash:     {flash['inference_time']:.2f}초")
        print(f"  개선도:    {speed_improvement:+.1f}% {'✅' if speed_improvement > 0 else '❌'}")
        
        # 메모리 비교
        memory_improvement = (std['peak_memory'] - flash['peak_memory']) / std['peak_memory'] * 100
        print(f"\\n💾 피크 메모리:")
        print(f"  Standard:  {std['peak_memory']:.2f}GB")
        print(f"  Flash:     {flash['peak_memory']:.2f}GB")
        print(f"  절약도:    {memory_improvement:+.1f}% {'✅' if memory_improvement > 0 else '❌'}")
        
        # 품질 비교
        print(f"\\n📝 출력 품질:")
        print(f"  Standard:  {std['output_length']} 문자, {std['tokens_generated']} 토큰")
        print(f"  Flash:     {flash['output_length']} 문자, {flash['tokens_generated']} 토큰")
        
        # 전체 평가
        print(f"\\n🎯 종합 평가:")
        if speed_improvement > 0 and memory_improvement > 0:
            print("  ✅ Flash Attention 2가 속도와 메모리 모두 개선")
        elif speed_improvement > 0:
            print("  ⚡ Flash Attention 2가 속도 개선 (메모리는 비슷)")
        elif memory_improvement > 0:
            print("  💾 Flash Attention 2가 메모리 절약 (속도는 비슷)")
        else:
            print("  ⚠️ Flash Attention 2 효과가 제한적")

def main():
    # 테스트할 이미지 파일 찾기
    image_path = None
    staging_dir = Path("staging/DICOM")
    if staging_dir.exists():
        image_files = list(staging_dir.glob("*.jpeg"))
        if image_files:
            image_path = image_files[0]  # 첫 번째 이미지 사용
    
    if not image_path or not image_path.exists():
        print("❌ 테스트용 이미지 파일을 찾을 수 없습니다.")
        print("   staging/DICOM/ 디렉토리에 이미지 파일이 있는지 확인하세요.")
        return
    
    print(f"🖼️ 테스트 이미지: {image_path}")
    
    tester = FlashAttentionTester()
    results = tester.run_comparison_test(image_path)
    
    print(f"\\n✅ Flash Attention 테스트 완료!")

if __name__ == "__main__":
    main()