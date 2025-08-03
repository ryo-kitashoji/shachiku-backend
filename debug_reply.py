#!/usr/bin/env python3
"""
返信生成デバッグスクリプト
"""
import asyncio
import sys
import os
import logging
from datetime import datetime, timezone

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from service.reply_generation.reply_service import ReplyService
from models.request_models import ReplyRequest, ReplySettings, ReplyMission, ReplyMessage

# 詳細ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_reply_generation():
    """返信生成のデバッグ"""
    print("=== 返信生成デバッグ開始 ===")
    
    # テストデータ作成（ユーザーの実際のリクエストと同じ）
    test_request = ReplyRequest(
        settings=ReplySettings(
            userId="user_123456",
            channel="chatwork:projectA",
            replyTo="上司の田中さん"
        ),
        mission=ReplyMission(
            instruction="相手の意見に共感しつつ距離を取る返信を作ってください。",
            goal="角を立てずにやんわり断ること"
        ),
        message=ReplyMessage(
            content="今日、飲みに行かない？",
            timestamp=datetime.fromisoformat("2025-08-05T14:23:00+09:00")
        )
    )
    
    try:
        # ReplyServiceインスタンス作成
        reply_service = ReplyService()
        print("ReplyServiceインスタンス作成完了")
        
        # 返信生成テスト
        result = await reply_service.generate_reply(
            request=test_request,
            max_length=512,
            temperature=0.7,
            top_p=0.9
        )
        
        print("=== 返信生成結果 ===")
        for key, value in result.items():
            print(f"{key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"デバッグ実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(debug_reply_generation())