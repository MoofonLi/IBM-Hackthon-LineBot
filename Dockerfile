FROM python:3.11.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    libfaiss-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


RUN mkdir -p docs


EXPOSE 5000


CMD ["python", "app.py"]