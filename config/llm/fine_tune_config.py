from dataclasses import dataclass
from typing import Optional


@dataclass
class FineTuneConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    dataset_path: str = "./data/training/excuses.jsonl"
    output_dir: str = "./data/models/fine_tuned"
    
    # トレーニングパラメータ
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    
    # LoRAパラメータ
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    
    # 評価設定
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    
    # ログ設定
    logging_dir: str = "./logs"
    logging_steps: int = 100
    
    # その他
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["c_attn", "c_proj"]


@dataclass 
class DatasetConfig:
    question_column: str = "question"
    answer_column: str = "excuse"
    max_length: int = 512
    train_test_split: float = 0.8
    validation_split: float = 0.1
    
    # データ前処理
    clean_text: bool = True
    remove_duplicates: bool = True
    min_length: int = 10
    max_examples: Optional[int] = None