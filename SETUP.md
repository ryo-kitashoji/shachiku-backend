# ShachikuAI セットアップガイド

## 📋 目次

1. [システム要件](#システム要件)
2. [環境構築](#環境構築)
3. [モデル設定](#モデル設定)
4. [トラブルシューティング](#トラブルシューティング)
5. [性能最適化](#性能最適化)

## システム要件

### 最小要件
- **OS**: Linux, macOS, Windows 10/11
- **Python**: 3.8以上（3.10推奨）
- **RAM**: 8GB以上
- **ストレージ**: 10GB以上の空き容量
- **ネットワーク**: インターネット接続（初回モデルダウンロード用）

### 推奨環境
- **RAM**: 16GB以上
- **GPU**: NVIDIA GPU（CUDA対応、4GB以上のVRAM）
- **ストレージ**: SSD 20GB以上

## 環境構築

### 1. Python環境の準備

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

#### macOS（Homebrew）
```bash
brew install python@3.10
```

#### Windows
[Python公式サイト](https://www.python.org/)からPython 3.10をダウンロードしてインストール

### 2. CUDAの設定（GPU使用の場合）

#### NVIDIA Driverの確認
```bash
nvidia-smi
```

#### CUDA Toolkitのインストール
```bash
# Ubuntu 20.04/22.04の場合
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 3. プロジェクトの準備

```bash
# リポジトリのクローン
git clone https://github.com/your-username/ShachikuAI.git
cd ShachikuAI

# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\\Scripts\\activate

# 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt
```

## モデル設定

### 1. デフォルト設定（推奨）

`.env`ファイルの設定:
```env
# 日本語GPT-1Bモデル（高品質、約5GB）
MODEL_NAME=rinna/japanese-gpt-1b
MODEL_PATH=./data/models/japanese-reply-model-1b
```

### 2. 軽量設定（リソース制限がある場合）

```env
# 日本語GPT-small（軽量、約400MB）
MODEL_NAME=rinna/japanese-gpt2-small
MODEL_PATH=./data/models/japanese-reply-model-small
```

### 3. 高性能設定（大容量メモリ環境）

```env
# 日本語GPT-neox（高性能、約10GB）
MODEL_NAME=rinna/bilingual-gpt-neox-4b
MODEL_PATH=./data/models/japanese-reply-model-neox
```

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. CUDA関連エラー
```
RuntimeError: CUDA out of memory
```

**解決方法:**
- GPUメモリが不足しています
- より軽量なモデルを使用する
- バッチサイズを小さくする
- CPUモードで実行する

```env
# CPUモードの設定
CUDA_VISIBLE_DEVICES=""
```

#### 2. モデルダウンロードエラー
```
ConnectionError: HTTPSConnectionPool
```

**解決方法:**
- ネットワーク接続を確認
- プロキシ設定が必要な場合は環境変数を設定
- 手動でモデルをダウンロード

```bash
# プロキシ設定例
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

#### 3. メモリ不足エラー
```
MemoryError: Unable to allocate array
```

**解決方法:**
- システムのRAMを増やす
- スワップファイルを設定
- より軽量なモデルを使用

```bash
# スワップファイル作成（Linux）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. 依存関係エラー
```
ImportError: No module named 'torch'
```

**解決方法:**
- 仮想環境が有効になっているか確認
- 依存関係の再インストール

```bash
pip install --force-reinstall -r requirements.txt
```

### ログの確認

```bash
# アプリケーションログ
tail -f logs/app.log

# システムログ（Linux）
journalctl -u docker -f
```

## 性能最適化

### 1. GPU最適化

```env
# Mixed Precision（メモリ節約）
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# CUDA最適化
CUDA_LAUNCH_BLOCKING=0
```

### 2. CPU最適化

```env
# マルチスレッド設定
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

### 3. キャッシュ最適化

```bash
# Hugging Faceキャッシュディレクトリの設定
export HF_HOME=/path/to/large/disk/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/large/disk/.cache/huggingface
```

### 4. Docker最適化

```yaml
# docker-compose.yml での設定
services:
  shachiku-ai:
    # 共有メモリサイズの増加
    shm_size: '2g'
    
    # CPUアフィニティの設定
    cpus: '4.0'
    
    # メモリ制限
    mem_limit: 16g
    memswap_limit: 16g
```

## 本番環境での考慮事項

### セキュリティ

1. **環境変数の管理**
```bash
# .envファイルの権限設定
chmod 600 .env
```

2. **APIキーの管理**
```bash
# 機密情報は環境変数で管理
export OPENAI_API_KEY="your-api-key"
```

### 監視

1. **ヘルスチェック**
```bash
# 定期的なヘルスチェック
curl -f http://localhost:8000/health || exit 1
```

2. **リソース監視**
```bash
# GPU使用率監視
nvidia-smi -l 1

# メモリ使用率監視
free -h
```

### スケーリング

1. **負荷分散**
```yaml
# docker-compose.yml
services:
  shachiku-ai:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 8G
```

2. **オートスケーリング**
```bash
# Kubernetesでのオートスケーリング設定
kubectl autoscale deployment shachiku-ai --cpu-percent=70 --min=1 --max=10
```

## サポート

問題が解決しない場合は、以下の情報を含めてIssueを作成してください：

1. OS・Python バージョン
2. エラーメッセージの全文
3. 設定ファイル（`.env`）の内容（機密情報を除く）
4. 実行したコマンド
5. システムリソース情報（`nvidia-smi`, `free -h`の出力）