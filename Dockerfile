FROM python:3.11.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    libfaiss-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 創建必要的目錄
RUN mkdir -p docs

# 暴露應用端口
EXPOSE 5000

# 設置啟動命令
CMD ["python", "app.py"]