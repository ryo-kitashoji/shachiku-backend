import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging
from config.llm.fine_tune_config import FineTuneConfig, DatasetConfig
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcuseFineTuner:
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def setup_model_and_tokenizer(self):
        logger.info(f"モデルとトークナイザーを初期化: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA設定を適用しました")
        
    def prepare_dataset(self, dataset_config: DatasetConfig):
        logger.info(f"データセットを準備: {self.config.dataset_path}")
        
        if not os.path.exists(self.config.dataset_path):
            logger.warning("トレーニングデータが見つかりません。サンプルデータを作成します。")
            self._create_sample_dataset()
        
        # JSONLファイルを読み込み
        data = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # データセット作成
        dataset = Dataset.from_list(data)
        
        # データセットを分割
        train_size = int(len(dataset) * dataset_config.train_test_split)
        val_size = int(len(dataset) * dataset_config.validation_split)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        
        # トークン化
        def tokenize_function(examples):
            prompts = [
                f"質問: {q}\n\n以下は上記の質問に対する丁寧で説得力のある言い訳です:\n\n{a}"
                for q, a in zip(examples[dataset_config.question_column], examples[dataset_config.answer_column])
            ]
            
            return self.tokenizer(
                prompts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"データセット準備完了 - 訓練: {len(self.train_dataset)}, 検証: {len(self.val_dataset)}")
    
    def _create_sample_dataset(self):
        sample_data = [
            {
                "question": "なぜ遅刻したのですか？",
                "excuse": "申し訳ございません、電車の遅延により到着が遅れてしまいました。今後は余裕を持って出発いたします。"
            },
            {
                "question": "なぜ宿題を忘れたのですか？",
                "excuse": "恐縮ですが、昨日体調を崩してしまい、十分な時間を確保できませんでした。明日までには必ず提出いたします。"
            },
            {
                "question": "なぜ会議に参加しなかったのですか？",
                "excuse": "申し上げにくいのですが、緊急の顧客対応が入り、どうしても抜けることができませんでした。議事録を確認させていただきます。"
            },
            {
                "question": "なぜ約束を破ったのですか？",
                "excuse": "ご迷惑をおかけして申し訳ございません、予期しない家庭の事情が発生し、やむを得ず予定を変更させていただきました。"
            }
        ]
        
        os.makedirs(os.path.dirname(self.config.dataset_path), exist_ok=True)
        with open(self.config.dataset_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"サンプルデータセットを作成: {self.config.dataset_path}")
    
    def train(self):
        logger.info("ファインチューニングを開始")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=self.config.dataloader_num_workers,
            fp16=True,
            report_to=None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        
        # モデルを保存
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"ファインチューニング完了。モデルを保存: {self.config.output_dir}")


def main():
    config = FineTuneConfig()
    dataset_config = DatasetConfig()
    
    fine_tuner = ExcuseFineTuner(config)
    fine_tuner.setup_model_and_tokenizer()
    fine_tuner.prepare_dataset(dataset_config)
    fine_tuner.train()


if __name__ == "__main__":
    main()