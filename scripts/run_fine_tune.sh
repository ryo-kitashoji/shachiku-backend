#!/bin/bash

echo "ShachikuAI ファインチューニングスクリプトを開始..."

cd "$(dirname "$0")/.."

if [ ! -f "requirements.txt" ]; then
    echo "エラー: requirements.txtが見つかりません"
    exit 1
fi

echo "依存関係をインストール中..."
pip install -r requirements.txt

echo "ファインチューニングデータを確認中..."
if [ ! -f "data/training/excuses.jsonl" ]; then
    echo "警告: トレーニングデータが見つかりません。サンプルデータが作成されます。"
fi

echo "ファインチューニングを開始..."
python scripts/fine_tuning/fine_tune.py

if [ $? -eq 0 ]; then
    echo "ファインチューニングが完了しました！"
    echo "モデルは data/models/fine_tuned に保存されています"
else
    echo "エラー: ファインチューニングに失敗しました"
    exit 1
fi