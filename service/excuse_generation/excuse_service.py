import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import logging
from client.llm.model_client import ModelClient

logger = logging.getLogger(__name__)


class ExcuseService:
    def __init__(self):
        self.model_client = ModelClient()
        self.excuse_prompts = [
            "申し訳ございません、",
            "すみません、実は",
            "恐縮ですが、",
            "申し上げにくいのですが、",
            "ご迷惑をおかけして申し訳ないのですが、"
        ]
        
    async def generate_excuse(
        self,
        question: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        try:
            prompt = self._create_excuse_prompt(question)
            
            response = await self.model_client.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            excuse_text = self._format_excuse(response["generated_text"])
            confidence = self._calculate_confidence(excuse_text)
            
            return {
                "text": excuse_text,
                "confidence": confidence,
                "prompt_used": prompt
            }
            
        except Exception as e:
            logger.error(f"言い訳生成中にエラー: {str(e)}")
            fallback_excuse = self._get_fallback_excuse(question)
            return {
                "text": fallback_excuse,
                "confidence": 0.3,
                "prompt_used": "fallback"
            }
    
    def _create_excuse_prompt(self, question: str) -> str:
        return f"""質問: {question}

以下は上記の質問に対する丁寧で説得力のある言い訳です:

"""
    
    def _format_excuse(self, generated_text: str) -> str:
        lines = generated_text.strip().split('\n')
        excuse_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith('質問:'):
                excuse_lines.append(line.strip())
        
        if excuse_lines:
            return excuse_lines[0]
        else:
            return "申し訳ございません、適切な対応ができませんでした。"
    
    def _calculate_confidence(self, excuse_text: str) -> float:
        confidence = 0.5
        
        if len(excuse_text) > 20:
            confidence += 0.2
        if any(polite in excuse_text for polite in ["申し訳", "すみません", "恐縮"]):
            confidence += 0.2
        if "ございます" in excuse_text or "でした" in excuse_text:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _get_fallback_excuse(self, question: str) -> str:
        import random
        fallback_excuses = [
            "申し訳ございません、システムの不具合により適切にお答えできませんでした。",
            "恐れ入ります、技術的な問題が発生しており、現在対応中です。",
            "すみません、予期しない事象が発生しており、調査を進めております。",
            "申し上げにくいのですが、現在システムメンテナンス中のため、正常な回答ができません。"
        ]
        return random.choice(fallback_excuses)