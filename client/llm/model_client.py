import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ModelClient:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        self.model_path = os.getenv("MODEL_PATH", "./data/models")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"モデルをロード中: {self.model_name}")
            
            if os.path.exists(os.path.join(self.model_path, "pytorch_model.bin")):
                logger.info(f"ローカルモデルを使用: {self.model_path}")
                model_path = self.model_path
            else:
                logger.info(f"Hugging Faceからモデルをダウンロード: {self.model_name}")
                model_path = self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"モデルのロードが完了 (デバイス: {self.device})")
            
        except Exception as e:
            logger.error(f"モデルロードエラー: {str(e)}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        try:
            if not self.pipeline:
                raise RuntimeError("モデルが初期化されていません")
            
            logger.info(f"テキスト生成開始: {prompt[:50]}...")
            
            generation_config = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            results = self.pipeline(prompt, **generation_config)
            
            if isinstance(results, list) and len(results) > 0:
                generated_text = results[0]["generated_text"]
            else:
                generated_text = "生成に失敗しました"
            
            logger.info(f"テキスト生成完了: {len(generated_text)} 文字")
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "config": generation_config
            }
            
        except Exception as e:
            logger.error(f"テキスト生成エラー: {str(e)}")
            return {
                "generated_text": "申し訳ございません、システムエラーが発生しました。",
                "prompt": prompt,
                "error": str(e)
            }
    
    def save_model(self, save_path: str):
        try:
            logger.info(f"モデルを保存中: {save_path}")
            os.makedirs(save_path, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info("モデルの保存が完了")
            
        except Exception as e:
            logger.error(f"モデル保存エラー: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "parameters": self.model.num_parameters() if self.model else None,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else None
        }