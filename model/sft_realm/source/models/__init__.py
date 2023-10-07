import logging

import transformers

from source.constants import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN 
from source.models.utils import smart_tokenizer_and_embedding_resize


def build_model(model_args, training_args):
    # Step 1: Initialize LLM
    logging.info(f"+ [Model] Initializing LM: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=None
    )

    # Step 2: Initialize tokenizer
    logging.info(f"+ [Model] Initializing Tokenizer: {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Step 3: Add special tokens
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    return model, tokenizer
