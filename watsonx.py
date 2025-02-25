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
        self.documents = []
    
    def process_documents(self, documents: List[Document]):
        """Process multiple documents and create unified index"""
        try:
            all_chunks = []
            
            # Process each document
            for doc in documents:
                #print(f"Processing document: {doc.metadata.get('source', 'unnamed')}")
                chunks = self._create_chunks(doc.content)
                #print(f"Created {len(chunks)} chunks")
                
                # Add source document info to each chunk
                for chunk in chunks:
                    all_chunks.append({
                        'content': chunk,
                        'metadata': doc.metadata
                    })
                
            self.chunks = all_chunks
            #print(f"Total chunks created: {len(all_chunks)}")
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk['content'] for chunk in all_chunks]
            #print("Generating embeddings...")
            embeddings = self.embedding_model.encode(chunk_texts)
            #print(f"Generated {len(embeddings)} embeddings")
            
            # Create FAISS index
            dimension = len(embeddings[0])
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(np.array(embeddings).astype('float32'))
            print(f"Created FAISS index with dimension {dimension}")
            
            return len(all_chunks)
            
        except Exception as e:
            print(f"Error in process_documents: {str(e)}")
            return 0
    
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
        """基於使用者查詢搜尋相關文件內容"""
        if not self.vector_store or not self.chunks:
            print("Warning: Vector store or chunks not initialized")  # Debug log
            return ""
                
        try:
            print(f"\nSearching for context related to: {query}")  # Debug log
            
            # Generate query vector
            query_vector = self.embedding_model.encode([query])
                
            # Search for most relevant document chunks
            distances, indices = self.vector_store.search(
                np.array(query_vector).astype('float32'), 
                top_k
            )
            
            print(f"Found {len(indices[0])} relevant chunks")  # Debug log
                
            # Get relevant document content with source info
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                chunk = self.chunks[idx]
                source = chunk['metadata'].get('source', 'unknown')
                content = chunk['content']
                similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity score
                # print(f"\nChunk {i+1} (similarity: {similarity:.2f}):")  # Debug log
                # print(f"Source: {source}")
                # print(f"Content: {content[:100]}...")  # Print first 100 chars
                relevant_chunks.append(f"來源: {source}\n相關度: {similarity:.2f}\n內容: {content}")
                
            return '\n\n---\n\n'.join(relevant_chunks)
                
        except Exception as e:
            print(f"Error in find_relevant_context: {str(e)}")
            return ""

    def generate_response(self, context: str, user_input: str, prompt_template: str, conversation_history: List[Dict] = None) -> Optional[str]:
        """使用指定的prompt模板生成回應"""
        # 組合歷史對話成文本
        history_text = ""
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    history_text += f"使用者: {msg['content']}\n"
                else:
                    history_text += f"助理: {msg['content']}\n"
        
        # 將歷史對話加入prompt
        prompt = prompt_template.format(
            context=context,
            user_input=user_input,
            conversation_history=history_text
        )

        payload = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 1000,
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
        