import requests
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from dataclasses import dataclass
import dotenv
import os


dotenv.load_dotenv()


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

class WatsonX:
    def __init__(self, api_key: str = None):
        """初始化 WatsonX API 和向量存儲"""
        # 如果沒有提供API金鑰，使用預設值
        if api_key is None:
            api_key = os.getenv("WATSONX_API_TOKEN")
            
        # IBM Cloud API 初始化
        token_response = requests.post(
            'https://iam.cloud.ibm.com/identity/token',
            data={
                "apikey": api_key,
                "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
            }
        )
        mltoken = token_response.json()["access_token"]
        
        self.url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mltoken}"
        }
        
        # 向量存儲初始化
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.vector_store = None
        self.chunks = []
        self.chunk_size = 300
        self.chunk_overlap = 30
        
    def process_document(self, content: str, metadata: Dict[str, Any] = None):
        """處理文檔並建立向量索引"""
        # 分批處理文檔
        self.chunks = self._create_chunks(content)
        
        # 批次生成嵌入向量
        batch_size = 32
        embeddings = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        # 初始化 FAISS 索引
        dimension = len(embeddings[0])
        self.vector_store = faiss.IndexFlatL2(dimension)
        
        # 添加向量到索引
        self.vector_store.add(np.array(embeddings).astype('float32'))
        
        return len(self.chunks)
    
    def _create_chunks(self, text: str) -> List[str]:
        """Smart document chunking with comprehensive error handling"""
        # Basic input validation
        if not text or not isinstance(text, str):
            print("Warning: Invalid input to _create_chunks")
            return []
            
        try:
            # Clean and validate text
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                print("Warning: Empty text after cleaning")
                return []
            
            # Initialize chunks list
            chunks = []
            
            # Section markers for medical documents
            section_markers = ['症狀：', '治療：', '注意事項：', '術後照護：', '##', '===']
            
            # Try splitting by section markers first
            has_sections = any(marker in text for marker in section_markers)
            if has_sections:
                for marker in section_markers:
                    if marker in text:
                        sections = text.split(marker)
                        for i, section in enumerate(sections):
                            if section and isinstance(section, str) and section.strip():
                                # Add marker back to all except first section
                                chunk = (marker + section.strip()) if i > 0 else section.strip()
                                chunks.append(chunk)
                
                if chunks:  # If we successfully created chunks using markers
                    return chunks
            
            # Fallback to sliding window if no sections found or sections empty
            text_length = len(text)
            start = 0
            
            while start < text_length:
                # Calculate end position with bounds checking
                end = min(start + self.chunk_size, text_length)
                if end <= start:
                    break
                    
                # Extract chunk with validation
                chunk = text[start:end]
                if isinstance(chunk, str) and chunk.strip():
                    chunks.append(chunk.strip())
                
                # Update start position with overlap
                start = max(end - self.chunk_overlap, start + 1)  # Ensure forward progress
                
            return chunks
            
        except Exception as e:
            print(f"Error in _create_chunks: {str(e)}")
            return []

    def find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """基於語義相似度搜索相關上下文"""
        if not self.vector_store or not self.chunks:
            return ""
            
        try:
            # 生成查詢向量
            query_vector = self.embedding_model.encode([query])
            
            # 執行向量搜索
            distances, indices = self.vector_store.search(
                np.array(query_vector).astype('float32'), 
                top_k
            )
            
            # 獲取最相關的文本片段
            relevant_chunks = [self.chunks[i] for i in indices[0]]
            
            return '\n\n'.join(relevant_chunks)
            
        except Exception as e:
            print(f"搜索錯誤: {str(e)}")
            return ""

    def generate_response(self, context: str, user_input: str, prompt_template: str) -> Optional[str]:
        """使用指定的prompt模板生成回應"""
        prompt = prompt_template.format(
            context=context,
            user_input=user_input
        )

        payload = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "min_new_tokens": 0,
                "repetition_penalty": 1
            },
            "model_id": "ibm/granite-3-8b-instruct",
            "project_id": "d91fb3ca-54ec-462a-9a26-491104a1d49d"
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            return response.json().get('results', [{}])[0].get(
                'generated_text',
                None
            )
            
        except requests.exceptions.RequestException as e:
            print(f"API 請求失敗: {str(e)}")
            return None