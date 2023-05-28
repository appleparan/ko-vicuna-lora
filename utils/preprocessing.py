import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers
from conversations import SeparatorStyle, get_conv_template
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conv_template("redpajama-incite_ko")
    roles = {
        "human": conv.roles[0],
        "gpt": conv.roles[1],
        "bard": conv.roles[1],
    }

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: Path,
    rng: np.random.Generator,
    _val_set_pct: int = -1,
) -> Dict:
    global local_rank

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank is None:
        local_rank = 0

    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")
    data_path_str = str(data_path.resolve())

    # Vicuna dataset doesn't fit into load_dataset
    # if data_path_str.endswith(".json") or data_path_str.endswith(".jsonl"):
    #     raw_data = load_dataset("json", data_files=data_path_str)
    # else:
    #     raw_data = load_dataset(data_path_str)

    raw_data = json.load(open(data_path_str, "r"))

    # Split train/test
    perm = rng.permutation(len(raw_data))
    if _val_set_pct < 0 or _val_set_pct >= 100:
        val_set_pct = 0.98
    else:
        val_set_pct = _val_set_pct / 100.0

    split = int(len(perm) * val_set_pct)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]

    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = SupervisedDataset(train_raw_data, tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Chat-7B-v0.1",
        model_max_length=2048,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_path = Path("../data/sharegpt_deepl_ko/ko_dataset_2.json")
    print(data_path.resolve())

    rng = np.random.default_rng(1234)

    print(make_supervised_data_module(tokenizer, data_path, rng))
