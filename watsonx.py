from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]

def get_iam_token(apikey: str) -> Optional[str]:
    """獲取 IBM Cloud IAM 令牌"""
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
        print(f"獲取令牌失敗: {response.status_code}, {response.text[:100]}")
    except Exception as e:
        print(f"獲取令牌時出錯: {str(e)}")
    return None

class WatsonX:
    def __init__(self, api_key: str = None):
        """初始化 WatsonX API 和向量存儲"""
        # 基本設置
        self.api_key = api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID", "your-project-id")
        self.model_id = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
        self.watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        
        # 獲取令牌
        self.iam_token = get_iam_token(self.api_key)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}"
        }
        
        # LLM 參數
        self.llm_params = {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        }
        
        # 初始化嵌入模型 - 使用支援繁體中文的多語言模型
        try:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("本地嵌入模型載入成功")
        except Exception as e:
            print(f"載入嵌入模型出錯: {str(e)}")
            self.embedding_model = None
            
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        # 儲存文檔和嵌入的空間
        self.chunks = []
        self.vector_store = None
        
    def process_documents(self, documents: List[Document]) -> int:
        """處理多個文檔並創建索引"""
        if not documents or not self.embedding_model:
            return 0
            
        try:
            # 文檔分塊
            all_texts = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.content)
                for chunk in chunks:
                    all_texts.append((chunk, doc.metadata))
                
            self.chunks = all_texts
            print(f"文檔分割為 {len(self.chunks)} 個塊")
            
            # 生成嵌入
            texts_only = [text for text, _ in self.chunks]
            embeddings = self.embedding_model.encode(texts_only)
            
            # 創建 FAISS 索引
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(np.array(embeddings).astype('float32'))
            
            return len(self.chunks)
            
        except Exception as e:
            print(f"處理文檔時出錯: {str(e)}")
            return 0
            
    def find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """搜索相關文檔內容"""
        if not self.vector_store or not self.chunks or not self.embedding_model:
            return ""
            
        try:
            # 生成查詢向量
            query_vector = self.embedding_model.encode([query])
            
            # 執行向量搜索
            distances, indices = self.vector_store.search(
                np.array(query_vector).astype('float32'), 
                top_k
            )
            
            # 整合搜索結果
            results = []
            for idx in indices[0]:
                if idx < len(self.chunks):
                    text, metadata = self.chunks[idx]
                    source = metadata.get('source', 'unknown')
                    results.append(f"來源: {source}\n內容: {text}")
            
            return '\n\n---\n\n'.join(results)
            
        except Exception as e:
            print(f"搜索相關上下文時出錯: {str(e)}")
            return ""
            
    def refresh_token(self):
        """刷新 IAM 令牌"""
        self.iam_token = get_iam_token(self.api_key)
        if self.iam_token:
            self.headers["Authorization"] = f"Bearer {self.iam_token}"
            return True
        return False
            
    def generate_response(self, context: str, user_input: str, prompt_template: str, conversation_history: List[Dict] = None) -> str:
        """使用 Llama 模型生成回應"""
        try:
            # 確保令牌有效
            if not self.iam_token:
                self.refresh_token()
                
            # 準備對話歷史
            history_text = ""
            if conversation_history:
                for msg in conversation_history[-3:]:  # 只使用最近3輪對話
                    role = "使用者" if msg["role"] == "user" else "助理"
                    history_text += f"{role}: {msg['content']}\n"
            
            # 使用模板格式化提示
            prompt = prompt_template.format(
                context=context[:1500] if context else "",  # 限制上下文長度
                user_input=user_input,
                conversation_history=history_text
            )
            
            # 呼叫 API 生成回應
            url = f"{self.watsonx_url}/ml/v1/text/generation?version=2023-05-29"
            
            payload = {
                "input": prompt,
                "parameters": self.llm_params,
                "model_id": self.model_id,
                "project_id": self.project_id
            }
            
            response = requests.post(
                url=url, 
                headers=self.headers,
                json=payload,
                timeout=15
            )
            
            # 處理 API 回應
            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
                    generated_text = data["results"][0].get("generated_text", "")
                    
                    # 清理回應內容
                    special_tokens = [
                        "<|begin_of_text|>", "<|end_of_text|>",
                        "<|start_header_id|>system<|end_header_id|>",
                        "<|start_header_id|>user<|end_header_id|>",
                        "<|start_header_id|>assistant<|end_header_id|>",
                        "<|eot_id|>"
                    ]
                    
                    for token in special_tokens:
                        generated_text = generated_text.replace(token, "")
                    
                    # 找到實際回應的開始
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
            
            # 如果 API 調用失敗
            print(f"API 調用失敗: {response.status_code}")
            return "您好！我是您的醫療諮詢顧問。請問有什麼可以幫您的嗎？"
            
        except Exception as e:
            print(f"生成回應時出錯: {str(e)}")
            return "您好！很抱歉，我暫時無法處理您的請求。請問有什麼其他問題嗎？"