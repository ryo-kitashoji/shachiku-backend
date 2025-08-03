# ShachikuAI - 言い訳生成・自動返信API

質問に対して丁寧で説得力のある言い訳を生成し、状況に応じた自動返信を提供するFastAPI + ローカルLLMアプリケーション

## 機能

- 質問に対する自然な言い訳の生成
- 状況に応じた自動返信の生成（断り、共感、距離を置く等）
- ローカルLLM（DeepSeek等）の利用
- ファインチューニング対応
- FastAPIベースのRESTful API
- Docker対応

## プロジェクト構造

```
ShachikuAI/
├── api/                    # APIレイヤー
│   └── v1/
│       ├── excuse_router.py
│       └── reply_router.py
├── client/                 # クライアントレイヤー
│   └── llm/
│       └── model_client.py
├── service/                # サービスレイヤー
│   ├── excuse_generation/
│   │   └── excuse_service.py
│   └── reply_generation/
│       └── reply_service.py
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

### 自動返信生成エンドポイント

```bash
curl -X POST "http://localhost:8000/shatiku-ai/generate-reply" \
     -H "Content-Type: application/json" \
     -d '{
       "settings": {
         "userId": "user123",
         "channel": "general",
         "replyTo": "user456"
       },
       "mission": {
         "instruction": "やんわり断る",
         "goal": "断る"
       },
       "message": {
         "content": "飲み会に参加しませんか？",
         "timestamp": "2024-01-01T12:00:00Z"
       }
     }'
```

レスポンス例：
```json
{
  "reply": "お忙しい中ご連絡いただきありがとうございます。申し訳ございませんが、今回は都合がつかないため参加が難しい状況です。またの機会がございましたら、ぜひよろしくお願いいたします。",
  "replyAt": "2024-01-01T12:00:00Z"
}
```

### ヘルスチェック

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/excuse/health
curl http://localhost:8000/shatiku-ai/health
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

### POST /shatiku-ai/generate-reply

自動返信を生成します。

**リクエスト:**
- `settings` (object, required): 設定情報
  - `userId` (string): ユーザーID
  - `channel` (string): チャンネル名
  - `replyTo` (string): 返信先ユーザーID
- `mission` (object, required): 返信のミッション
  - `instruction` (string): 指示内容（例: "やんわり断る", "共感", "距離"）
  - `goal` (string): 目標（例: "断る"）
- `message` (object, required): メッセージ情報
  - `content` (string): メッセージ内容
  - `timestamp` (datetime): タイムスタンプ

**レスポンス:**
- `reply` (string): 生成された返信
- `replyAt` (datetime): 返信時刻

### GET /shatiku-ai/health

自動返信サービスのヘルスチェックを行います。

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