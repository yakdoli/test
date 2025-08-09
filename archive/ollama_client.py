"""
Xinference API를 통해 이미지를 마크다운으로 변환하는 모듈
"""
import requests
import json
import base64
import time
from pathlib import Path
from typing import List, Optional
import config

class XinferenceClient:
    def __init__(self):
        self.base_url = config.XINFERENCE_BASE_URL
        self.model_name = config.XINFERENCE_MODEL_NAME
        self.model_uid = config.XINFERENCE_MODEL_UID
        
    def encode_image_to_base64(self, image_path: Path) -> str:
        """
        이미지 파일을 base64로 인코딩
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            str: base64 인코딩된 이미지 데이터
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def check_xinference_connection(self) -> bool:
        """
        Xinference 서버 연결 상태 확인
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def check_model_availability(self) -> bool:
        """
        지정된 모델이 사용 가능한지 확인하고 model_uid 설정
        
        Returns:
            bool: 모델 사용 가능 여부
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json().get('data', [])
                for model in models:
                    if model.get('id', '').startswith(self.model_name):
                        self.model_uid = model.get('id')
                        return True
            return False
        except requests.exceptions.RequestException:
            return False
    
    def get_syncfusion_prompt(self) -> str:
        """
        Syncfusion SDK 매뉴얼에 특화된 프롬프트 생성
        
        Returns:
            str: LLM 미세조정 및 RAG에 최적화된 프롬프트
        """
        return """Convert this Syncfusion SDK documentation image to structured markdown format optimized for LLM fine-tuning and RAG applications.

CRITICAL REQUIREMENTS:

## Code Processing
- Extract ALL code snippets with proper language identification
- Preserve exact syntax, indentation, and formatting
- Use appropriate code blocks with language tags (```csharp, ```vb, ```xml, etc.)
- Maintain complete method signatures, parameter lists, and return types
- Include inline code elements using backticks for class names, properties, methods

## API Documentation Structure
- Identify and properly format: Classes, Namespaces, Methods, Properties, Events, Enums
- Use consistent heading hierarchy (# for main topics, ## for classes, ### for methods)
- Create clear parameter tables with: Name | Type | Description | Default Value
- Document return values with type and description
- Extract exception information if present

## Technical Content Enhancement
- Preserve all technical terminology exactly as written
- Maintain version-specific information and compatibility notes
- Include performance considerations and best practices
- Extract configuration settings and their valid values
- Document dependencies and required assemblies

## Structured Output Format
- Use descriptive headers that include class/namespace context
- Create linkable anchors for cross-references
- Format examples with clear "Example:" or "Usage:" headers
- Include "See Also" sections for related APIs
- Add metadata comments for categorization

## Content Completeness
- Extract ALL visible text without omission
- Preserve table structures with proper markdown formatting
- Maintain numbered/bulleted lists with correct nesting
- Include notes, warnings, and tips in appropriate callout format
- Capture image captions and figure references

## RAG Optimization
- Use semantic section breaks for better chunking
- Include contextual keywords for improved searchability
- Maintain hierarchical relationships between parent/child concepts
- Add implicit context where beneficial for standalone understanding

Focus on creating documentation that serves as high-quality training data for LLM fine-tuning while being immediately useful for RAG retrieval systems."""
    
    def convert_image_to_markdown(self, image_path: Path) -> Optional[str]:
        """
        이미지를 마크다운 텍스트로 변환 (Xinference API 사용)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Optional[str]: 변환된 마크다운 텍스트 또는 None
        """
        max_retries = 3
        timeout = 120

        for attempt in range(max_retries):
            try:
                image_base64 = self.encode_image_to_base64(image_path)
                
                # Xinference OpenAI 호환 API 사용
                payload = {
                    "model": self.model_uid or self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.get_syncfusion_prompt()
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4000,
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    print(f"❌ API 요청 실패 (시도 {attempt + 1}/{max_retries}): {response.status_code}")
                    print(f"   응답: {response.text}")
            
            except requests.exceptions.Timeout:
                print(f"⏰ 타임아웃 (시도 {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"❌ 이미지 변환 실패 ({image_path.name}, 시도 {attempt + 1}/{max_retries}): {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(2)

        return None
    
    def convert_images_to_markdown(self, image_paths: List[Path]) -> str:
        """
        여러 이미지를 순서대로 마크다운으로 변환하고 결합
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            str: 결합된 마크다운 텍스트
        """
        markdown_content = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"🔄 페이지 {i}/{len(image_paths)} 변환 중: {image_path.name}")
            
            page_markdown = self.convert_image_to_markdown(image_path)
            
            if page_markdown:
                # 페이지 구분자 추가
                if i > 1:
                    markdown_content.append("\n---\n")
                
                markdown_content.append(f"<!-- 페이지 {i} -->\n")
                markdown_content.append(page_markdown)
                print(f"✅ 페이지 {i} 변환 완료")
            else:
                print(f"❌ 페이지 {i} 변환 실패")
                markdown_content.append(f"\n<!-- 페이지 {i} 변환 실패 -->\n")
        
        return "\n".join(markdown_content)
    
    def post_process_syncfusion_content(self, markdown_content: str, pdf_name: str) -> str:
        """
        Syncfusion SDK 매뉴얼 콘텐츠 후처리
        
        Args:
            markdown_content: 원본 마크다운 콘텐츠
            pdf_name: PDF 파일명
            
        Returns:
            str: 후처리된 마크다운 콘텐츠
        """
        if not config.SYNCFUSION_MODE:
            return markdown_content
            
        processed_content = []
        
        # 메타데이터 헤더 추가
        if config.INCLUDE_METADATA:
            metadata = f"""---
title: "{pdf_name} - Syncfusion SDK Documentation"
type: "api-documentation"
framework: "syncfusion"
version: "v11"
extracted_date: "{Path(__file__).stat().st_mtime}"
optimized_for: ["llm-training", "rag-retrieval"]
---

"""
            processed_content.append(metadata)
        
        # 의미 단위 청킹을 위한 섹션 마커 추가
        if config.SEMANTIC_CHUNKING:
            lines = markdown_content.split('\n')
            in_code_block = False
            
            for i, line in enumerate(lines):
                # 코드 블록 상태 추적
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                
                # 주요 섹션 구분점에 청킹 마커 추가
                if not in_code_block and (
                    line.startswith('# ') or 
                    line.startswith('## ') and ('class' in line.lower() or 'namespace' in line.lower())
                ):
                    if i > 0:  # 첫 번째 헤더가 아닌 경우
                        processed_content.append('\n<!-- CHUNK_BOUNDARY -->\n')
                
                processed_content.append(line)
        else:
            processed_content.append(markdown_content)
        
        return '\n'.join(processed_content)
    
    def extract_code_snippets(self, markdown_content: str, pdf_name: str) -> dict:
        """
        마크다운에서 코드 스니펫 추출 및 분류
        
        Args:
            markdown_content: 마크다운 콘텐츠
            pdf_name: PDF 파일명
            
        Returns:
            dict: 언어별 코드 스니펫 딕셔너리
        """
        if not config.EXTRACT_CODE_SNIPPETS:
            return {}
            
        import re
        
        code_snippets = {
            'csharp': [],
            'vb': [],
            'xml': [],
            'javascript': [],
            'css': [],
            'other': []
        }
        
        # 코드 블록 패턴 매칭
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, markdown_content, re.DOTALL)
        
        for lang, code in matches:
            lang = lang.lower() if lang else 'other'
            
            # 언어 매핑
            if lang in ['c#', 'csharp', 'cs']:
                code_snippets['csharp'].append(code.strip())
            elif lang in ['vb', 'vb.net', 'visualbasic']:
                code_snippets['vb'].append(code.strip())
            elif lang in ['xml', 'xaml']:
                code_snippets['xml'].append(code.strip())
            elif lang in ['js', 'javascript']:
                code_snippets['javascript'].append(code.strip())
            elif lang == 'css':
                code_snippets['css'].append(code.strip())
            else:
                code_snippets['other'].append(code.strip())
        
        return code_snippets

# 호환성을 위한 별칭
OllamaClient = XinferenceClient

if __name__ == "__main__":
    client = XinferenceClient()
    
    # 연결 테스트
    if client.check_xinference_connection():
        print("✅ Xinference 서버 연결 성공")
        
        if client.check_model_availability():
            print(f"✅ 모델 '{client.model_name}' 사용 가능 (UID: {client.model_uid})")
        else:
            print(f"❌ 모델 '{client.model_name}' 사용 불가능")
    else:
        print("❌ Xinference 서버 연결 실패")