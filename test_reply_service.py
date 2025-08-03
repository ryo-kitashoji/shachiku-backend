#!/usr/bin/env python3
"""
reply_service.pyのテストスクリプト
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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_reply_generation():
    """返信生成のテスト"""
    logger.info("=== 返信生成テスト開始 ===")
    
    # テストデータ作成
    test_request = ReplyRequest(
        settings=ReplySettings(
            userId="test_user",
            channel="general",
            replyTo="田中さん"
        ),
        mission=ReplyMission(
            instruction="やんわりと断る",
            goal="角が立たないように断る"
        ),
        message=ReplyMessage(
            content="明日の飲み会に参加しませんか？",
            timestamp=datetime.now(timezone.utc)
        )
    )
    
    try:
        # ReplyServiceインスタンス作成
        reply_service = ReplyService()
        logger.info("ReplyServiceインスタンス作成完了")
        
        # 返信生成テスト
        result = await reply_service.generate_reply(
            request=test_request,
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info("=== 返信生成結果 ===")
        logger.info(f"返信: {result['reply']}")
        logger.info(f"生成時刻: {result['replyAt']}")
        logger.info(f"使用プロンプト: {result['prompt_used']}")
        
        if 'confidence' in result:
            logger.info(f"信頼度: {result['confidence']}")
        
        if 'raw_generation' in result:
            logger.info(f"生成テキスト（raw）: {result['raw_generation']}")
            
        if 'error' in result:
            logger.error(f"エラー: {result['error']}")
            
        if 'ai_error' in result:
            logger.error(f"AI生成エラー: {result['ai_error']}")
        
        return result
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_multiple_scenarios():
    """複数のシナリオでテスト"""
    logger.info("=== 複数シナリオテスト開始 ===")
    
    scenarios = [
        {
            "name": "断るシナリオ",
            "instruction": "やんわりと断る",
            "goal": "角が立たないように断る",
            "message": "明日の飲み会に参加しませんか？"
        },
        {
            "name": "共感シナリオ",
            "instruction": "共感を示す",
            "goal": "相手の気持ちに寄り添う",
            "message": "最近忙しくて疲れています"
        },
        {
            "name": "距離を置くシナリオ",
            "instruction": "適切な距離を保つ",
            "goal": "プロフェッショナルな関係を維持",
            "message": "今度一緒にランチしませんか？"
        }
    ]
    
    reply_service = ReplyService()
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\n--- シナリオ {i}: {scenario['name']} ---")
        
        test_request = ReplyRequest(
            settings=ReplySettings(
                userId="test_user",
                channel="general",
                replyTo="同僚"
            ),
            mission=ReplyMission(
                instruction=scenario["instruction"],
                goal=scenario["goal"]
            ),
            message=ReplyMessage(
                content=scenario["message"],
                timestamp=datetime.now(timezone.utc)
            )
        )
        
        result = await reply_service.generate_reply(test_request)
        logger.info(f"メッセージ: {scenario['message']}")
        logger.info(f"返信: {result['reply']}")
        logger.info(f"プロンプト種類: {result['prompt_used']}")
        
        if 'confidence' in result:
            logger.info(f"信頼度: {result['confidence']:.2f}")

if __name__ == "__main__":
    print("ShachikuAI 返信生成テスト")
    print("=" * 50)
    
    # 基本テスト
    asyncio.run(test_reply_generation())
    
    print("\n" + "=" * 50)
    
    # 複数シナリオテスト
    asyncio.run(test_multiple_scenarios())
    
    print("\nテスト完了")