import copy
import json
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
import numpy as np
from torch.utils.data import Dataset

from source.utils import default_conversation
from source.constants import IGNORE_INDEX, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = InstructionDataset
    raw_dataset = json.load(open(data_args.data_path, "r")) 
    raw_dataset = _prepro_data_dict(raw_dataset)

    # Split train/test
    np.random.seed(125)
    perm = np.random.permutation(len(raw_dataset))
    split = int(len(perm) * 0.95)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_dataset[i] for i in train_indices]
    eval_raw_data = [raw_dataset[i] for i in eval_indices]
    print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class InstructionDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(InstructionDataset, self).__init__()
        logging.info("Loading data...")
        self.tokenizer = tokenizer
        self.list_data_dict = raw_data

    def __len__(self):
        return len(self.list_data_dict) 

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources] 
        data_dict = preprocess(copy.deepcopy([e["conversations"] for e in sources]), self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add symbol '### ' at the beginning each sentence, with end symbol '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    conversations = []
    intermediates = []
    for source in sources: 
        header = f"{default_conversation.system}" 
        conversation, intermediate = _add_speaker_and_symbol(header, source)
        conversations.append(conversation) 
        intermediates.append(intermediate) 
       
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
  
    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool) 
        for inter in inters: 
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1 
            end_idx = tokenized["input_ids"][1].size(0) 
            mask[start_idx:end_idx] = True 
        target[~mask] = IGNORE_INDEX 
    return dict(input_ids=input_ids, labels=targets)

def _add_speaker_and_symbol(header, source, get_conversation=True) :
    """Add speaker and start/end symbol on each round."""
    BEGIN_symbol = DEFAULT_BOS_TOKEN
    END_symbol = DEFAULT_EOS_TOKEN
    conversation = header
    intermediate = [] 
    for sentence in source: 
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
       
        value = (from_str + ": " + BEGIN_symbol + sentence["value"] + END_symbol)
        if sentence["from"].lower() == "gpt":
            start = conversation + from_str + ": " + BEGIN_symbol
            end = conversation + value
            intermediate.append([start, end]) 
        if get_conversation:
            conversation += value
    return conversation, intermediate 

def _prepro_data_dict(list_data_dict):
    list_data_dict = [item for item in list_data_dict if len(item["conversations"]) > 0]
    return list_data_dict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [ 
        tokenizer(
            text,
            return_tensors="pt", 
            padding="longest", 
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings 
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids, 
        labels=labels, 
        input_ids_lens=input_ids_lens, 
        labels_lens=labels_lens, 
    )
