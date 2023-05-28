import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers

# from conversations import SeparatorStyle, get_conv_template
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from utils.conversations import SeparatorStyle, get_conv_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


local_rank = 0


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conv_template("polyglot_ko")
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

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        cur_len = 0
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
            z = torch.where(z == IGNORE_TOKEN_ID, 0, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    # f" {conversation}"
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
        "EleutherAI/polyglot-ko-3.8b",
        model_max_length=2048,
        use_fast=True,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )

    # tokenizer.pad_token = tokenizer.unk_token
    data_path = Path("../data/sharegpt_deepl_ko/ko_dataset_2.json")
    print(data_path.resolve())

    rng = np.random.default_rng(1234)

    sequence_en = str("Hello, I'm an assistant.")
    sequence_kr = str("<bot>(<봇>): 테스트 완료.")
    sequence_kr = str("<human>(<사람>): 테스트 완료.")
    new_tokens = ["<bot>(<봇>)", "<human>(<사람>)"]

    print("EleutherAI/polyglot-ko-3.8b")
    tokenizer.add_tokens(list(new_tokens))
    # model.resize_token_embeddings(len(tokenizer))
    print(tokenizer.tokenize(sequence_en))
    encoded = tokenizer(sequence_kr)
    # input_ids = tokenizer(
    #     sequence,
    #     return_tensors="pt",
    #     padding="max_length",
    #     max_length=tokenizer.model_max_length,
    #     truncation=True,
    # ).input_ids
    print([tokenizer.decode([id]) for id in encoded.input_ids])
    # print(make_supervised_data_module(tokenizer, data_path, rng))
    raw_data = [
        {
            "id": "kdRPRhv_95",
            "conversations": [
                {
                    "from": "gpt",
                    "value": "죄송하지만 하인즈가 등장한 모든 영화 목록을 가지고 있지는 않습니다. 하지만 하인즈 제품이 영화나 TV 프로그램에서 사람들이 일상적으로 사용하는 흔한 아이템으로 자주 등장한다는 점은 말씀드릴 수 있습니다. 예를 들어, 하인즈 케첩으로 샌드위치를 만드는 캐릭터나 음식 접시 위에 하인즈 그레이비를 붓는 캐릭터를 볼 수 있습니다. 브랜드와 제품은 시청자가 알아볼 수 있고 공감할 수 있어야 하기 때문에 주방, 식당, 레스토랑을 배경으로 하는 장면에 자주 등장합니다.",
                },
                {"from": "human", "value": "하인즈는 중요한 역사적 사건에 참석했나요?"},
                {
                    "from": "gpt",
                    "value": "제가 알기로는 하인즈 제품이 다양한 역사적 사건과 행사에 사용되었을 수 있지만, 특정 사건에서 중요한 역할을 했다는 구체적인 사례나 증거는 찾을 수 없었습니다. 하인즈 제품은 다양한 기간 동안 많은 사람들의 일상 생활의 일부였으며 중요한 순간과 행사에서 소비되었을 가능성이 있습니다. 그러나 더 많은 정보나 구체적인 사례가 없으면 확실하게 말하기는 어렵습니다.",
                },
                {"from": "human", "value": "예를 들어 주시겠어요?"},
                {
                    "from": "gpt",
                    "value": "하인즈 제품이 중요한 역사적 사건에 사용되었다는 구체적인 사례는 알지 못합니다. 그러나 다른 많은 식품과 마찬가지로 하인즈 제품도 역사적으로 다양한 시기에 소비되었을 가능성이 있습니다.",
                },
            ],
        },
        {
            "id": "Lizj058_0",
            "conversations": [
                {
                    "from": "human",
                    "value": "브라우저 선택 API에 대해 알고 싶습니다. 개요부터 시작하겠습니다.",
                },
                {
                    "from": "gpt",
                    "value": "브라우저 선택 API는 개발자가 브라우저 창이나 문서 내에서 선택한 텍스트에 액세스하고 조작할 수 있는 애플리케이션 프로그래밍 인터페이스(API)입니다. 이 API는 개발자가 사용자 선택에 실시간으로 반응할 수 있도록 하여 보다 인터랙티브하고 동적인 웹 애플리케이션을 만들 수 있는 방법을 제공합니다. 개발자는 선택한 텍스트의 콘텐츠를 수정, 검색, 복사할 수 있으며, 강조 표시 또는 주석 추가와 같은 작업을 수행할 수 있습니다. 이 API는 텍스트 편집기나 워드 프로세서와 같은 텍스트 기반 웹 애플리케이션을 만드는 데 필수적입니다.",
                },
                {
                    "from": "human",
                    "value": "웹 텍스트 편집기에서 이 API를 사용하여 구현할 수 있는 작업의 예를 제공해 주시겠어요?",
                },
                {
                    "from": "gpt",
                    "value": "예. 웹 기반 텍스트 편집기를 만들고 사용자가 텍스트를 강조 표시할 수 있도록 하려고 한다고 가정해 보겠습니다. 브라우저 선택 API를 사용하면 `mouseup` 이벤트를 수신하여 사용자가 텍스트를 선택할 때를 감지한 다음 `getSelection()` 메서드를 사용하여 선택한 텍스트를 가져올 수 있습니다. 그런 다음 다음 자바스크립트 코드에 표시된 것처럼 텍스트를 강조 표시하는 CSS 클래스를 추가하여 텍스트를 수정할 수 있습니다:```document.addEventListener('mouseup', function() {  var selection = window.getSelection();  if (selection.toString().length) {    var range = selection.getRangeAt(0);    var span = document.createElement('span');    span.className = 'highlight';    range.surroundContents(span);  }});```이 코드는 문서에서 `mouseup` 이벤트를 수신하고 텍스트가 선택되었는지 확인합니다. 텍스트가 선택된 경우 `getRangeAt(0)`을 사용하여 선택 범위를 가져옵니다. 그런 다음 새 `span` 요소를 생성하고 `highlight` 클래스를 추가한 다음 `surroundContents()` 메서드를 사용하여 선택한 텍스트를 둘러쌉니다.그런 다음 `highlight` 클래스를 CSS에서 정의하여 강조 표시된 텍스트에 다양한 스타일을 적용할 수 있습니다. 다음은 예시입니다:```.highlight {  background-color: yellow;  color: black;}```이 코드를 사용하면 사용자가 편집기에서 텍스트를 선택할 때마다 노란색 배경과 검은색 텍스트로 강조 표시됩니다.",
                },
            ],
        },
    ]
    print(SupervisedDataset(raw_data, tokenizer=tokenizer)[0])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "huggyllama/llama-7b"
    )
    print("huggyllama/llama-7b")
    print(tokenizer.tokenize(sequence_en))
    print(tokenizer.tokenize(sequence_kr))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "kykim/bert-kor-base"
    )
    print("kykim/bert-kor-base")
    print(tokenizer.tokenize(sequence_en))
    print(tokenizer.tokenize(sequence_kr))
