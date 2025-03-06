from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

def get_iam_token(apikey: str) -> Optional[str]:
    """Get IBM Cloud IAM token"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "apikey": apikey,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        if response.status_code == 200:
            return response.json().get("access_token")
        print(f"{response.status_code}, {response.text[:100]}")
    except Exception as e:
        print(f"{str(e)}")
    return None

class WatsonX:
    def __init__(self, api_key: str = None):
        """Initialize WatsonX API"""

        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.model_id = os.getenv("WATSONX_MODEL_ID")
        self.watsonx_url = os.getenv("WATSONX_URL")
        
        # Get Token
        self.iam_token = get_iam_token(self.api_key)
        self.token_timestamp = time.time() if self.iam_token else 0
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}" if self.iam_token else ""
        }
        
        # LLM Parameters
        self.llm_params = {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        }
        
        # Embedding Model
        try:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Success")
        except Exception as e:
            print(f"Error: {str(e)}")
            self.embedding_model = None
            
        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        self.chunks = []
        self.vector_store = None
    
    def is_token_expired(self):
        """Check if token is expired (55 min threshold)"""
        if not self.iam_token or not self.token_timestamp:
            return True
        return (time.time() - self.token_timestamp) > (55 * 60)  
        
    def refresh_token(self):
        """Refresh IAM token"""
        print("Refreshing IAM token...")
        new_token = get_iam_token(self.api_key)
        if new_token:
            self.iam_token = new_token
            self.token_timestamp = time.time()
            self.headers["Authorization"] = f"Bearer {self.iam_token}"
            print("Token refreshed")
            return True
        print("Token refresh failed")
        return False
            
    def process_documents(self, documents: List[Document]) -> int:
        """Process documents and create index"""
        if not documents or not self.embedding_model:
            return 0
            
        try:
            # Split documents
            all_texts = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.content)
                for chunk in chunks:
                    all_texts.append((chunk, doc.metadata))
                
            self.chunks = all_texts
            print(f"Documents split into {len(self.chunks)} chunks")
            
            # Generate embeddings
            texts_only = [text for text, _ in self.chunks]
            embeddings = self.embedding_model.encode(texts_only)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(np.array(embeddings).astype('float32'))
            
            return len(self.chunks)
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return 0
            
    def find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Search for relevant document content"""
        if not self.vector_store or not self.chunks or not self.embedding_model:
            return ""
            
        try:
            # Generate query vector
            query_vector = self.embedding_model.encode([query])
            
            # Perform vector search
            distances, indices = self.vector_store.search(
                np.array(query_vector).astype('float32'), 
                top_k
            )
            
            # Integrate search results
            results = []
            for idx in indices[0]:
                if idx < len(self.chunks):
                    text, metadata = self.chunks[idx]
                    source = metadata.get('source', 'unknown')
                    results.append(f"來源: {source}\n內容: {text}")
            
            return '\n\n---\n\n'.join(results)
            
        except Exception as e:
            print(f"Error searching context: {str(e)}")
            return ""
            
    def generate_response(self, context: str, user_input: str, prompt_template: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using LLM"""
        try:
            # Check if token needs refresh
            if not self.iam_token or self.is_token_expired():
                print("Token missing or expired, refreshing...")
                if not self.refresh_token():
                    return "很抱歉，無法連接到服務。請稍後再試。"
                
            # Prepare conversation history
            history_text = ""
            if conversation_history:
                for msg in conversation_history[-3:]:  # Use only last 3 turns
                    role = "使用者" if msg["role"] == "user" else "助理"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Format prompt with template
            prompt = prompt_template.format(
                context=context[:1500] if context else "",  # Limit context length
                user_input=user_input,
                conversation_history=history_text
            )
            
            # Call API to generate response
            url = f"{self.watsonx_url}/ml/v1/text/generation?version=2023-05-29"
            
            payload = {
                "input": prompt,
                "parameters": self.llm_params,
                "model_id": self.model_id,
                "project_id": self.project_id
            }
            
            max_retries = 2
            for attempt in range(max_retries):
                response = requests.post(
                    url=url, 
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                # Process API response
                if response.status_code == 200:
                    data = response.json()
                    if "results" in data and data["results"]:
                        generated_text = data["results"][0].get("generated_text", "")
                        
                        # Clean response content
                        special_tokens = [
                            "<|begin_of_text|>", "<|end_of_text|>",
                            "<|start_header_id|>system<|end_header_id|>",
                            "<|start_header_id|>user<|end_header_id|>",
                            "<|start_header_id|>assistant<|end_header_id|>",
                            "<|eot_id|>"
                        ]
                        
                        for token in special_tokens:
                            generated_text = generated_text.replace(token, "")
                                
                        # Find actual response start
                        if "您是一位" in generated_text or "請遵守以下準則" in generated_text:
                            lines = generated_text.split('\n')
                            processed_lines = []
                            skip_lines = True
                            
                            for line in lines:
                                if skip_lines and ("使用者:" in line or "助理:" in line or 
                                                "歷史對話:" in line or "準則:" in line):
                                    continue
                                else:
                                    skip_lines = False
                                    if line.strip():
                                        processed_lines.append(line)
                            
                            generated_text = '\n'.join(processed_lines)
                        
                        return generated_text.strip() or "您好，請問有什麼可以幫您的嗎？"
                        
                # Handle token expiration
                elif response.status_code == 401:
                    print(f"Token expired (attempt {attempt+1}/{max_retries}), refreshing...")
                    if self.refresh_token() and attempt < max_retries - 1:
                        continue  # Retry request
                    else:
                        print("Token refresh failed or max retries reached")
                        break
                else:
                    print(f"API call failed: {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"Retrying {attempt+2}/{max_retries}...")
                        continue
                    break
            
            # If all attempts fail
            return "您好！很抱歉，我暫時無法連接到服務。請稍後再試。"
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "您好！很抱歉，我暫時無法處理您的請求。請問有什麼其他問題嗎？"