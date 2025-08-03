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
        self.model_client = ModelClient()
        
    async def generate_reply(
        self,
        request: ReplyRequest,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        try:
            logger.info("AIを使用して返信を生成開始")
            
            # プロンプトを作成
            prompt = self._create_reply_prompt(request)
            logger.info(f"生成したプロンプト: {prompt[:100]}...")
            
            # モデルクライアントを使ってテキスト生成
            # max_new_tokensを使用して適切な生成量を指定
            max_new_tokens = 80  # 返信として十分な長さ
            logger.info(f"プロンプト文字数: {len(prompt)}")
            logger.info(f"生成パラメータ: max_new_tokens={max_new_tokens}, temperature=0.8, top_p=0.9")
            
            generation_result = await self.model_client.generate_text(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            
            if "error" in generation_result:
                logger.warning(f"AI生成でエラー、フォールバックを使用: {generation_result['error']}")
                fallback_reply = self._get_fallback_reply(request)
                return {
                    "reply": fallback_reply,
                    "replyAt": datetime.now(timezone.utc),
                    "prompt_used": "fallback",
                    "ai_error": generation_result["error"]
                }
            
            # 生成されたテキストをフォーマット
            generated_text = generation_result["generated_text"]
            formatted_reply = self._format_reply(generated_text)
            
            # デバッグモードでのみ詳細ログを出力
            if os.getenv("DEBUG_MODE", "false").lower() == "true":
                logger.info(f"生成されたrawテキスト: '{generated_text}'")
                logger.info(f"フォーマット後の返信: '{formatted_reply}'")
            
            confidence_score = self._calculate_confidence(formatted_reply)
            
            reply_at = datetime.now(timezone.utc)
            
            logger.info(f"AI返信生成完了: {formatted_reply[:50]}...")
            
            # 環境変数でデバッグモードを制御
            debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
            
            result = {
                "reply": formatted_reply,
                "replyAt": reply_at,
                "prompt_used": "ai_generated",
                "confidence": confidence_score
            }
            
            # デバッグ情報を追加（開発環境のみ）
            if debug_mode:
                result.update({
                    "debug": {
                        "raw_generation": generated_text[:200],
                        "prompt": prompt[:100] + "...",
                        "generation_config": generation_result.get("config", {})
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error(f"自動返信生成中にエラー: {str(e)}")
            fallback_reply = self._get_fallback_reply(request)
            return {
                "reply": fallback_reply,
                "replyAt": datetime.now(timezone.utc),
                "prompt_used": "fallback",
                "error": str(e)
            }
    
    def _create_reply_prompt(self, request: ReplyRequest) -> str:
        # 具体的な例を含むプロンプト
        instruction = request.mission.instruction
        message = request.message.content
        sender = request.settings.replyTo
        
        if "共感" in instruction and "距離" in instruction:
            example = "ありがとうございます。お誘いいただき嬉しいのですが、今回は都合がつかず参加が難しいです。"
        elif "断る" in instruction:
            example = "申し訳ございませんが、今回は参加が難しいです。"
        elif "共感" in instruction:
            example = "お疲れ様です。その通りですね。"
        else:
            example = "ありがとうございます。検討いたします。"
        
        return f"""「{message}」という{sender}からのメッセージに対して、{instruction}という方針で返信してください。

例: {example}

返信:"""
    
    def _format_reply(self, generated_text: str) -> str:
        # 生成されたテキストをクリーンアップ
        text = generated_text.strip()
        
        # 最初の2-3文を組み合わせて適切な返信を作成
        sentences = text.split('。')
        valid_sentences = []
        
        for sentence in sentences[:3]:  # 最初の3文まで
            sentence = sentence.strip()
            if (sentence and 
                len(sentence) > 5 and 
                not sentence.startswith(('引用', '田中さんからの', '上司からの')) and
                '返信' not in sentence and
                'メッセージ' not in sentence and
                'ポイント' not in sentence):
                valid_sentences.append(sentence)
        
        if valid_sentences:
            # 適切な長さの返信を作成
            reply = valid_sentences[0]
            if len(valid_sentences) > 1 and len(reply) < 30:
                reply += '。' + valid_sentences[1]
            
            # 長すぎる場合は調整
            if len(reply) > 60:
                reply = reply[:60] + '...'
            
            # 句点を確実に追加
            if not reply.endswith('。'):
                reply += '。'
            
            return reply
        
        # フォールバック
        return "ありがとうございます。検討させていただきます。"
    
    def _calculate_confidence(self, reply_text: str) -> float:
        """返信テキストの信頼度を計算"""
        confidence = 0.5  # ベーススコア
        
        # 長さによる評価
        if 10 <= len(reply_text) <= 150:
            confidence += 0.2
        elif len(reply_text) < 10:
            confidence -= 0.3
        
        # 敬語の使用チェック
        polite_expressions = ['です', 'ます', 'ございます', 'いたします', 'させて', 'お疲れ様']
        if any(expr in reply_text for expr in polite_expressions):
            confidence += 0.2
        
        # エラーメッセージでないかチェック
        if "申し訳ございません、適切な返信を生成できませんでした" in reply_text:
            confidence = 0.1
        
        # 不自然な文字列がないかチェック
        if '【' in reply_text or '】' in reply_text:
            confidence -= 0.2
        
        # 0.0-1.0の範囲に収める
        return max(0.0, min(1.0, confidence))
    
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