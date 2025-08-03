# 軽量なPython 3.10イメージを使用
FROM python:3.10-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依存関係ファイルをコピーして先にインストール（キャッシュ効率化）
COPY requirements.txt .

# Pythonパッケージをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 必要なディレクトリを作成
RUN mkdir -p data/models data/training logs

# 環境変数を設定
ENV PYTHONPATH=/app
ENV MODEL_NAME=rinna/japanese-gpt-1b
ENV MODEL_PATH=./data/models/japanese-reply-model-1b
ENV MAX_LENGTH=512
ENV TEMPERATURE=0.7
ENV TOP_P=0.9
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV FINE_TUNE_ENABLED=true
ENV FINE_TUNE_DATA_PATH=./data/training/excuses.jsonl

# ポートを公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーション起動
CMD ["python", "main.py"]