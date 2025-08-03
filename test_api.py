#!/usr/bin/env python3
"""
APIエンドポイントのテストスクリプト
"""
import requests
import json
from datetime import datetime

def test_reply_api():
    """返信生成APIのテスト"""
    base_url = "http://localhost:8000"
    
    # テストデータ
    test_data = {
        "settings": {
            "userId": "test_user",
            "channel": "general", 
            "replyTo": "田中さん"
        },
        "mission": {
            "instruction": "やんわりと断る",
            "goal": "角が立たないように断る"
        },
        "message": {
            "content": "明日の飲み会に参加しませんか？",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        print("APIテスト開始...")
        
        # ヘルスチェック
        health_response = requests.get(f"{base_url}/shatiku-ai/health")
        print(f"ヘルスチェック: {health_response.status_code} - {health_response.json()}")
        
        # 返信生成API呼び出し
        response = requests.post(
            f"{base_url}/shatiku-ai/generate-reply",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"API応答ステータス: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("=== API返信生成結果 ===")
            print(f"返信: {result.get('reply', 'N/A')}")
            print(f"生成時刻: {result.get('replyAt', 'N/A')}")
        else:
            print(f"エラー: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("エラー: APIサーバーに接続できません。サーバーが起動しているか確認してください。")
    except Exception as e:
        print(f"テスト実行エラー: {str(e)}")

if __name__ == "__main__":
    test_reply_api()