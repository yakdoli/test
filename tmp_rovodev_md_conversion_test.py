#!/usr/bin/env python3
"""
qwen2.5vl을 통한 마크다운 변환 테스트
"""
import time
import sys
from pathlib import Path
import config
from ollama_client import OllamaClient
from main import PDFToMarkdownConverter
import random

class MarkdownConversionTester:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.staging_dir = config.STAGING_DIR
        self.output_dir = config.OUTPUT_DIR
        
    def test_ollama_connection(self):
        """Ollama 연결 및 모델 테스트"""
        print("🔍 Ollama 연결 테스트")
        print("-" * 40)
        
        # 서버 연결 확인
        if not self.ollama_client.check_ollama_connection():
            print("❌ Ollama 서버에 연결할 수 없습니다.")
            print("   해결 방법: ollama serve 실행")
            return False
        
        print("✅ Ollama 서버 연결 성공")
        
        # 모델 사용 가능 여부 확인
        if not self.ollama_client.check_model_availability():
            print(f"❌ 모델 '{config.OLLAMA_MODEL}'을 사용할 수 없습니다.")
            print(f"   해결 방법: ollama pull {config.OLLAMA_MODEL}")
            return False
        
        print(f"✅ 모델 '{config.OLLAMA_MODEL}' 사용 가능")
        return True
    
    def select_test_images(self, doc_name, max_images=3):
        """테스트용 이미지 선택"""
        staging_path = self.staging_dir / doc_name
        
        if not staging_path.exists():
            print(f"❌ {doc_name} 스테이징 디렉토리가 존재하지 않습니다.")
            return []
        
        image_files = list(staging_path.glob(f"*.{config.IMAGE_FORMAT.lower()}"))
        
        if not image_files:
            print(f"❌ {doc_name}에 이미지 파일이 없습니다.")
            return []
        
        # 첫 페이지, 중간 페이지, 마지막 페이지 선택
        selected_images = []
        
        if len(image_files) >= 1:
            selected_images.append(image_files[0])  # 첫 페이지
        
        if len(image_files) >= 3:
            mid_index = len(image_files) // 2
            selected_images.append(image_files[mid_index])  # 중간 페이지
        
        if len(image_files) >= 2:
            selected_images.append(image_files[-1])  # 마지막 페이지
        
        # 최대 개수 제한
        return selected_images[:max_images]
    
    def test_single_image_conversion(self, image_path):
        """단일 이미지 마크다운 변환 테스트"""
        print(f"🔄 이미지 변환 테스트: {image_path.name}")
        
        start_time = time.time()
        
        try:
            # 마크다운 변환
            result = self.ollama_client.convert_image_to_markdown(image_path)
            
            elapsed = time.time() - start_time
            
            if result:
                print(f"✅ 변환 성공 ({elapsed:.1f}초)")
                print(f"   결과 길이: {len(result)} 문자")
                
                # 결과 미리보기 (처음 200자)
                preview = result[:200].replace('\n', ' ') + "..." if len(result) > 200 else result
                print(f"   미리보기: {preview}")
                
                return {
                    'success': True,
                    'time': elapsed,
                    'length': len(result),
                    'content': result
                }
            else:
                print(f"❌ 변환 실패 ({elapsed:.1f}초)")
                return {
                    'success': False,
                    'time': elapsed,
                    'error': 'Empty result'
                }
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 변환 오류 ({elapsed:.1f}초): {str(e)}")
            return {
                'success': False,
                'time': elapsed,
                'error': str(e)
            }
    
    def test_document_sample(self, doc_name, max_images=3):
        """문서 샘플 테스트"""
        print(f"\n📄 {doc_name} 문서 샘플 테스트")
        print("=" * 60)
        
        # 테스트 이미지 선택
        test_images = self.select_test_images(doc_name, max_images)
        
        if not test_images:
            return None
        
        print(f"선택된 이미지: {len(test_images)}개")
        
        results = []
        total_time = 0
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n[{i}/{len(test_images)}] {image_path.name}")
            result = self.test_single_image_conversion(image_path)
            results.append(result)
            total_time += result['time']
        
        # 결과 요약
        success_count = sum(1 for r in results if r['success'])
        avg_time = total_time / len(results) if results else 0
        
        print(f"\n📊 {doc_name} 테스트 결과:")
        print(f"   성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print(f"   평균 시간: {avg_time:.1f}초/페이지")
        print(f"   총 소요 시간: {total_time:.1f}초")
        
        return {
            'doc_name': doc_name,
            'total_images': len(test_images),
            'success_count': success_count,
            'total_time': total_time,
            'avg_time': avg_time,
            'results': results
        }
    
    def test_multiple_documents(self, doc_names):
        """여러 문서 테스트"""
        print("🚀 다중 문서 마크다운 변환 테스트")
        print("=" * 80)
        
        all_results = []
        
        for doc_name in doc_names:
            result = self.test_document_sample(doc_name, max_images=2)  # 각 문서당 2개 이미지
            if result:
                all_results.append(result)
        
        # 전체 결과 요약
        if all_results:
            print(f"\n🎉 전체 테스트 결과 요약")
            print("=" * 80)
            
            total_success = sum(r['success_count'] for r in all_results)
            total_tested = sum(r['total_images'] for r in all_results)
            total_time = sum(r['total_time'] for r in all_results)
            avg_time = sum(r['avg_time'] for r in all_results) / len(all_results)
            
            print(f"테스트된 문서: {len(all_results)}개")
            print(f"테스트된 이미지: {total_tested}개")
            print(f"전체 성공률: {total_success}/{total_tested} ({total_success/total_tested*100:.1f}%)")
            print(f"평균 변환 시간: {avg_time:.1f}초/페이지")
            print(f"총 소요 시간: {total_time:.1f}초")
            
            # 문서별 성능 비교
            print(f"\n📊 문서별 성능:")
            for result in sorted(all_results, key=lambda x: x['avg_time']):
                print(f"   {result['doc_name']:<20} {result['avg_time']:>6.1f}초/페이지 "
                      f"({result['success_count']}/{result['total_images']} 성공)")
        
        return all_results
    
    def test_full_document_conversion(self, doc_name):
        """전체 문서 변환 테스트"""
        print(f"\n🔄 {doc_name} 전체 문서 변환 테스트")
        print("=" * 60)
        
        try:
            converter = PDFToMarkdownConverter()
            
            print(f"변환 시작: {doc_name}")
            start_time = time.time()
            
            # 전체 변환 실행
            success = converter.run(doc_name)
            
            elapsed = time.time() - start_time
            
            if success:
                print(f"✅ {doc_name} 전체 변환 성공 ({elapsed/60:.1f}분)")
                
                # 결과 파일 확인
                output_file = config.OUTPUT_DIR / f"{doc_name}.md"
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"   출력 파일: {output_file}")
                    print(f"   파일 크기: {file_size:,} bytes")
                    
                    # 내용 미리보기
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')[:10]  # 처음 10줄
                        print(f"   내용 미리보기:")
                        for line in lines:
                            if line.strip():
                                print(f"      {line}")
                
                return True
            else:
                print(f"❌ {doc_name} 전체 변환 실패")
                return False
                
        except Exception as e:
            print(f"❌ {doc_name} 변환 오류: {str(e)}")
            return False

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "sample"
    
    tester = MarkdownConversionTester()
    
    # Ollama 연결 확인
    if not tester.test_ollama_connection():
        print("\n❌ Ollama 환경이 준비되지 않았습니다.")
        return 1
    
    if test_type == "sample":
        # 샘플 테스트 (여러 문서의 일부 페이지)
        test_docs = ["DICOM", "Gauge", "PDF Viewer", "common"]
        tester.test_multiple_documents(test_docs)
        
    elif test_type == "single":
        # 단일 문서 전체 변환
        doc_name = sys.argv[2] if len(sys.argv) > 2 else "DICOM"
        tester.test_full_document_conversion(doc_name)
        
    elif test_type == "quick":
        # 빠른 테스트 (1개 문서, 1개 이미지)
        doc_name = sys.argv[2] if len(sys.argv) > 2 else "DICOM"
        result = tester.test_document_sample(doc_name, max_images=1)
        
        if result and result['success_count'] > 0:
            print(f"\n✅ 빠른 테스트 성공!")
            print(f"   변환 시간: {result['avg_time']:.1f}초/페이지")
            print(f"   전체 문서 예상 시간: {result['avg_time'] * 12 / 60:.1f}분 (DICOM 기준)")
        else:
            print(f"\n❌ 빠른 테스트 실패")
    
    else:
        print("사용법:")
        print("  python md_conversion_test.py sample           # 샘플 테스트")
        print("  python md_conversion_test.py single DICOM     # 단일 문서 전체 변환")
        print("  python md_conversion_test.py quick DICOM      # 빠른 테스트")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())