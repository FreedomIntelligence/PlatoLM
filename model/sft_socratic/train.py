import os
import pathlib
from typing import Optional
from dataclasses import dataclass, field

import wandb
import torch
import transformers
from transformers import Trainer
from peft import get_peft_model_state_dict

from source.models import build_model
from source.utils import safe_save_model_for_hf_trainer
from source.datasets.datasets import make_supervised_data_module

# replace it with your own info.
# os.environ["WANDB_API_KEY"]='your key'
# wandb.init(
#     project="",
#     entity="",
# )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="FreedomIntelligence/Socratic-7B")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = build_model(model_args, training_args)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
