# ShachikuAI - 言い訳生成API

質問に対して丁寧で説得力のある言い訳を生成するFastAPI + ローカルLLMアプリケーション

## 機能

- 質問に対する自然な言い訳の生成
- ローカルLLM（DeepSeek等）の利用
- ファインチューニング対応
- FastAPIベースのRESTful API
- Docker対応

## プロジェクト構造

```
ShachikuAI/
├── api/                    # APIレイヤー
│   └── v1/
│       └── excuse_router.py
├── client/                 # クライアントレイヤー
│   └── llm/
│       └── model_client.py
├── service/                # サービスレイヤー
│   └── excuse_generation/
│       └── excuse_service.py
├── models/                 # データモデル
│   └── request_models.py
├── config/                 # 設定ファイル
│   └── llm/
│       └── fine_tune_config.py
├── scripts/                # スクリプト
│   └── fine_tuning/
│       ├── fine_tune.py
│       └── run_fine_tune.sh
├── data/                   # データ
│   ├── training/           # トレーニングデータ
│   └── models/             # モデル保存先
├── tests/                  # テストファイル
│   ├── unit/
│   └── integration/
├── main.py                 # メインアプリケーション
├── requirements.txt        # 依存関係
├── Dockerfile
├── docker-compose.yml
└── .env                    # 環境変数
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを編集して設定を調整：

```env
MODEL_NAME=microsoft/DialoGPT-medium
MODEL_PATH=./data/models
MAX_LENGTH=512
TEMPERATURE=0.7
TOP_P=0.9
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. APIの起動

```bash
python main.py
```

または

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Docker での起動

```bash
docker-compose up --build
```

## API利用方法

### 言い訳生成エンドポイント

```bash
curl -X POST "http://localhost:8000/v1/excuse/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "なぜ遅刻したのですか？",
       "max_length": 512,
       "temperature": 0.7,
       "top_p": 0.9
     }'
```

レスポンス例：
```json
{
  "question": "なぜ遅刻したのですか？",
  "excuse": "申し訳ございません、電車の遅延により到着が遅れてしまいました。今後は余裕を持って出発いたします。",
  "confidence": 0.9
}
```

### ヘルスチェック

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/excuse/health
```

## ファインチューニング

### 1. トレーニングデータの準備

`data/training/excuses.jsonl`に以下の形式でデータを追加：

```json
{"question": "質問文", "excuse": "言い訳文"}
```

### 2. ファインチューニングの実行

```bash
./scripts/run_fine_tune.sh
```

または

```bash
python scripts/fine_tuning/fine_tune.py
```

### 3. ファインチューニング後のモデル利用

`.env`ファイルを更新：

```env
MODEL_NAME=./data/models/fine_tuned
```

## API仕様

### POST /v1/excuse/generate

言い訳を生成します。

**リクエスト:**
- `question` (string, required): 質問文
- `max_length` (int, optional): 最大生成長 (default: 512)
- `temperature` (float, optional): 生成の創造性 (default: 0.7)
- `top_p` (float, optional): 核サンプリング (default: 0.9)

**レスポンス:**
- `question` (string): 入力された質問
- `excuse` (string): 生成された言い訳
- `confidence` (float): 信頼度スコア

## 開発

### テストの実行

```bash
pytest tests/
```

### ログの確認

```bash
tail -f logs/app.log
```

## ライセンス

MIT License