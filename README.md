# ShachikuAI - 言い訳生成・自動返信API

質問に対して丁寧で説得力のある言い訳を生成し、状況に応じた自動返信を提供するFastAPI + ローカルLLMアプリケーション

## 🚀 新機能

✨ **AI返信生成機能を実装しました！**
- 日本語GPT-1Bモデル（rinna/japanese-gpt-1b）による高品質な返信生成
- やんわり断る、共感する、距離を保つなど、様々な対応方針に対応
- フォールバック機能付きで安定動作

## 機能

- 📝 **言い訳生成**: 質問に対する自然で説得力のある言い訳を生成
- 🤖 **AI自動返信**: 状況に応じた適切な返信を生成（断り、共感、距離を置く等）
- 🧠 **ローカルLLM**: 日本語対応の高品質なモデル（rinna/japanese-gpt-1b）
- 🔧 **ファインチューニング対応**: カスタムデータでのモデル改善
- 🌐 **FastAPI**: 高性能なRESTful API
- 🐳 **Docker対応**: 簡単なデプロイとスケーリング
- 📊 **信頼度スコア**: 生成された返信の品質評価

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

## 🛠️ セットアップ

### 前提条件

- Python 3.8+
- CUDA対応GPU（推奨、CPUでも動作可能）
- 8GB以上のRAM
- 10GB以上の空きディスク容量

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/ShachikuAI.git
cd ShachikuAI
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env`ファイルを編集（初回起動時に自動的にモデルがダウンロードされます）：

```env
# モデル設定
MODEL_NAME=rinna/japanese-gpt-1b
MODEL_PATH=./data/models/japanese-reply-model-1b

# 生成パラメータ
MAX_LENGTH=512
TEMPERATURE=0.7
TOP_P=0.9

# API設定
API_HOST=0.0.0.0
API_PORT=8000

# ファインチューニング設定
FINE_TUNE_ENABLED=true
FINE_TUNE_DATA_PATH=./data/training/excuses.jsonl
```

### 5. ディレクトリ構造の準備

```bash
mkdir -p data/models data/training
```

### 6. APIの起動

```bash
# 開発モード（自動リロード付き）
python main.py

# または uvicorn で直接起動
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# API ドキュメントにアクセス
open http://localhost:8000/docs
```

### 8. Docker での起動（オプション）

```bash
docker-compose up --build
```

## ⚠️ 重要な注意事項

1. **初回起動時**: 日本語GPT-1Bモデル（約5GB）が自動的にダウンロードされます
2. **メモリ要件**: GPUメモリ4GB以上推奨、CPUの場合は8GB以上のRAM
3. **モデルファイル**: `.gitignore`によりモデルファイルはGitで管理されません
4. **セキュリティ**: 本番環境では`.env`ファイルの権限設定に注意してください

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