import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import logging
from datetime import datetime, timezone
from client.llm.model_client import ModelClient
from models.request_models import ReplyRequest

logger = logging.getLogger(__name__)


class ReplyService:
    def __init__(self):
        self.model_client = None
        
    async def generate_reply(
        self,
        request: ReplyRequest,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        try:
            logger.info("フォールバック機能を使用して返信を生成")
            fallback_reply = self._get_fallback_reply(request)
            reply_at = datetime.now(timezone.utc)
            
            return {
                "reply": fallback_reply,
                "replyAt": reply_at,
                "prompt_used": "fallback"
            }
            
        except Exception as e:
            logger.error(f"自動返信生成中にエラー: {str(e)}")
            fallback_reply = self._get_fallback_reply(request)
            return {
                "reply": fallback_reply,
                "replyAt": datetime.now(timezone.utc),
                "prompt_used": "fallback"
            }
    
    def _create_reply_prompt(self, request: ReplyRequest) -> str:
        return f"""## 返信生成指示

### 設定情報
- ユーザーID: {request.settings.userId}
- チャンネル: {request.settings.channel}
- 返信相手: {request.settings.replyTo}

### ミッション
- 指示: {request.mission.instruction}
- 目標: {request.mission.goal}

### 受信メッセージ
- 内容: {request.message.content}
- 受信日時: {request.message.timestamp}

### 返信要求
上記の情報を元に、ミッションに沿った適切な返信メッセージを日本語で生成してください。
返信は丁寧で自然な日本語とし、相手との関係性や状況を考慮した内容にしてください。

返信:
"""
    
    def _format_reply(self, generated_text: str) -> str:
        lines = generated_text.strip().split('\n')
        reply_lines = []
        
        found_reply_section = False
        for line in lines:
            if line.strip() == "返信:" or found_reply_section:
                found_reply_section = True
                if line.strip() and line.strip() != "返信:":
                    reply_lines.append(line.strip())
        
        if reply_lines:
            return reply_lines[0]
        else:
            return "申し訳ございません、適切な返信を生成できませんでした。"
    
    def _get_fallback_reply(self, request: ReplyRequest) -> str:
        instruction = request.mission.instruction.lower()
        goal = request.mission.goal.lower()
        
        if "断る" in goal or "やんわり" in instruction:
            return "お忙しい中ご連絡いただきありがとうございます。申し訳ございませんが、今回は都合がつかないため参加が難しい状況です。またの機会がございましたら、ぜひよろしくお願いいたします。"
        elif "共感" in instruction:
            return "お疲れ様です。おっしゃる通りですね。大変参考になるご意見をありがとうございます。"
        elif "距離" in instruction:
            return "ご連絡いただきありがとうございます。検討させていただき、改めてご連絡いたします。"
        else:
            return "お疲れ様です。ご連絡いただきありがとうございます。内容を確認させていただき、適切に対応いたします。"