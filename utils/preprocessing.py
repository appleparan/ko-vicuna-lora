import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers
from utils.conversations import SeparatorStyle, get_conv_template
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
        "EleutherAI/polyglot-ko-5.8b",
        model_max_length=2048,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        use_fast=True,
    )
    # tokenizer.pad_token = tokenizer.unk_token
    data_path = Path("../data/sharegpt_deepl_ko/ko_dataset_2.json")
    print(data_path.resolve())

    rng = np.random.default_rng(1234)

    sequence_kr = str("<조수> : 테스트 완료.")
    sequence_kr = str("<bot>(<봇>): 테스트 완료.")
    sequence_kr = str("<Human>(<사람>): 테스트 완료.")
    sequence_en = str("Hello, I'm an assistant.")
    new_tokens = ["<bot>(<봇>)", "<human>(<사람>)"]

    print("EleutherAI/polyglot-ko-5.8b")
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
            "id": "QWJhYvA_0",
            "conversations": [
            {
                "from": "human",
                "value": "고객을 위해 이러한 전략과 전술을 구현하는 성장 마케팅 대행사와 관련하여 Jeff Walker의 제품 출시 공식의 주요 아이디어를 글머리 기호로 요약하세요..."
            },
            {
                "from": "gpt",
                "value": "다음은 성장 마케팅 대행사가 고객을 위해 적용할 수 있는 제프 워커의 제품 출시 공식의 주요 아이디어입니다:\r\n\r\n1. 타겟 고객과 그들의 니즈를 파악합니다: 제품이나 서비스의 이상적인 고객을 파악하고 이들에게 공감을 불러일으킬 수 있는 메시지를 작성합니다.\r\n2. 사전 출시: 입소문을 만들고, 사용 후기 및 사례 연구를 수집하고, 소셜 미디어를 사용하여 인지도를 높여 출시에 대한 기대감과 흥미를 조성합니다.\r\n3. 출시: 잘 짜여진 출시 시퀀스를 사용하여 판매와 전환을 극대화하세요. 여기에는 보너스 제공, 희소성 창출, 마감일을 사용하여 긴박감을 조성하는 것이 포함될 수 있습니다.\r\n4. 출시 후: 고객과 후속 조치를 취하고, 피드백을 수집하고, 지속적으로 가치를 제공하여 고객의 참여와 충성도를 유지하세요.\r\n5. 제품군을 만듭니다: 초기 제품이 성공하면 동일한 고객의 다른 요구 사항을 해결하는 추가 제품 또는 서비스를 만들어 제품을 확장합니다.\r\n6. 지속적인 최적화: 출시 프로세스와 제품군을 지속적으로 모니터링하고 최적화하여 결과를 개선합니다.\r\n7. 커뮤니티 구축: 출시 프로세스를 사용하여 제품에 대한 열정이 있고 널리 알리는 데 도움을 줄 수 있는 고객 커뮤니티를 구축하세요.\r\n8. 자동화를 활용하세요: 기술 및 자동화를 사용하여 출시 프로세스를 간소화하고 효율성을 개선하세요."
            },
            {
                "from": "human",
                "value": "고객을 위해 이러한 전략과 전술을 구현하는 성장 마케팅 대행사와 관련하여 브렌든 버처드의 전문가 아카데미의 주요 아이디어를 글머리 기호로 요약하세요..."
            },
            {
                "from": "gpt",
                "value": "다음은 성장 마케팅 대행사가 고객을 위해 적용할 수 있는 브렌든 버처드 전문가 아카데미의 주요 아이디어입니다:\r\n\r\n1. 전문성을 파악하라: 자신의 전문 분야를 정의하고 다른 사람과 차별화되는 고유한 가치 제안을 만드세요.\r\n2. 콘텐츠 전략 수립: 자신의 전문성을 보여주고 청중에게 가치를 제공할 수 있는 콘텐츠 전략을 개발하세요. 여기에는 블로그 게시물, 동영상, 팟캐스트, 소셜 미디어 콘텐츠가 포함될 수 있습니다.\r\n3. 오디언스 구축하기: 소셜 미디어, 광고 및 기타 채널을 사용하여 잠재고객을 확보하고 가시성을 높입니다.\r\n4. 제품 개발: 전문 지식에 부합하고 잠재 고객의 요구를 충족하는 제품과 서비스를 만드세요. 여기에는 디지털 제품, 강좌, 코칭 및 컨설팅 서비스가 포함될 수 있습니다.\r\n5. 브랜드 만들기: 전문성을 반영하고 청중의 공감을 불러일으킬 수 있는 강력한 브랜드를 개발하세요.\r\n6. 권위 구축하기: 이벤트에서 연설하거나 책을 출판하거나 팟캐스트나 TV 프로그램에 출연하여 업계에서 권위자로 자리매김하세요.\r\n7. 전문 지식을 수익화하세요: 전문 지식과 제품을 활용하여 수익을 창출하는 수익화 전략을 개발하세요.\r\n8. 팀을 구성하세요: 비즈니스가 성장함에 따라 비즈니스를 확장하고 고객에게 우수한 서비스를 제공할 수 있도록 전문가로 구성된 팀을 구성하세요.\r\n9. 지속적인 개선: 제품, 서비스, 마케팅 전략을 지속적으로 개선하여 경쟁에서 앞서 나가고 고객에게 최상의 경험을 제공하세요.\r\n10. 영향력에 집중하세요: 궁극적으로 고객의 삶에 긍정적인 영향을 미치고 전문성을 활용하여 세상을 변화시키는 데 집중하세요."
            },
            {
                "from": "human",
                "value": "제프 워커의 제품 출시 공식과 \"출시\" 책에 나오는 정신적 트리거는 무엇인가요?"
            },
            {
                "from": "gpt",
                "value": "다음은 Jeff Walker의 제품 출시 공식과 \"출시\" 책에 설명된 정신적 트리거입니다:\r\n\r\n1. 희소성: 출시되는 제품이나 서비스의 가용성을 제한하여 긴박감을 조성합니다.\r\n2. 권위: 업계에서 전문가 또는 권위자로서의 입지를 구축하여 청중에게 신뢰와 믿음을 구축합니다.\r\n3. 사회적 증거: 추천글, 사례 연구 및 기타 형태의 사회적 증거를 사용하여 제품 또는 서비스의 효과와 가치를 입증합니다.\r\n4. 상호성: 판매 또는 약정을 요청하기 전에 잠재 고객에게 가치 있는 것을 제공하는 것.\r\n5. 호감도: 잠재 고객과 개인적인 관계를 구축하고 호감 가는 페르소나를 만들어 판매 가능성을 높입니다.\r\n6. 기대감: 출시에 앞서 기대감과 흥분을 조성하여 입소문을 내고 제품 또는 서비스에 대한 관심을 높입니다.\r\n7. 독점성: 특정 그룹에게 조기 액세스 또는 특별 거래를 제공하여 독점적인 느낌을 줍니다.\r\n8. 소속감: 브랜드에 대한 참여도와 충성도를 높이기 위해 청중들 사이에 커뮤니티 또는 소속감을 조성합니다.\r\n9. 스토리텔링: 스토리텔링을 사용하여 정서적 수준에서 청중과 연결하고 제품 또는 서비스를 더 관련성 있고 기억에 남도록 만듭니다.\r\n\r\n이러한 정신적 트리거를 활용하여 기업은 타겟 고객의 공감을 불러일으키고 매출과 성장을 촉진하는 효과적인 제품 출시 전략을 수립할 수 있습니다."
            },
            {
                "from": "human",
                "value": "에서 희소성과 긴급성이 가장 강력한 정신적 유발 요인이며 지난 8년 동안 가장 성과가 좋았던 캠페인의 원동력이었던 이유를 설명합니다. 오니 로빈스는 인간의 행동과 동기를 이끄는 6가지 핵심 욕구에 대해 설명합니다. 이 여섯 가지 욕구는 다음과 같습니다:\r\n\r\n1. 확실성: 안전, 안정성, 예측 가능성에 대한 욕구. 여기에는 편안함, 보안, 환경에 대한 통제에 대한 욕구가 포함됩니다.\r\n2. 다양성: 참신함, 흥분, 변화에 대한 욕구. 여기에는 모험, 자극, 새로운 경험에 대한 욕구가 포함됩니다.\r\n3. 중요성: 중요성: 중요하고 특별하며 독특하다고 느끼고자 하는 욕구. 여기에는 인정, 성취, 존경에 대한 욕구가 포함됩니다.\r\n4. 연결: 사회적 연결과 사랑에 대한 욕구. 여기에는 친밀감, 소속감, 우정에 대한 욕구가 포함됩니다.\r\n5. 성장: 성장: 개인적인 성장과 발전의 욕구. 여기에는 학습, 자기계발, 발전의 욕구가 포함됩니다.\r\n6. 기여: 세상에 긍정적인 영향을 미치고 싶은 욕구. 여기에는 환원하고, 다른 사람을 돕고, 변화를 만들고자 하는 욕구가 포함됩니다.\r\n\r\n토니 로빈스에 따르면 이 여섯 가지 욕구는 보편적인 것으로 모든 개인에게 적용됩니다. 사람마다 이러한 욕구의 우선순위는 다를 수 있고 충족하는 방식도 다를 수 있지만, 이러한 욕구는 인간 행동의 근본적인 동인입니다. 이러한 욕구를 이해함으로써 개인은 자신의 동기와 행동에 대한 통찰력을 얻을 수 있으며, 이러한 지식을 활용하여 보다 만족스럽고 의미 있는 삶을 만들 수 있습니다."
            },
            {
                "from": "gpt",
                "value": "희소성과 긴박감은 제품이나 서비스에 대한 독점성과 중요도를 만들어 내기 때문에 마케팅에서 가장 강력한 정신적 자극을 주는 두 가지 요소입니다. 한정된 수량이나 프로모션 기한을 설정하여 고객이 너무 늦기 전에 행동을 취하고 구매하도록 동기를 부여할 수 있습니다.\r\n\r\n지난 8년 동안 많은 기업이 이러한 트리거를 마케팅 캠페인에 활용하여 큰 성공을 거두었습니다. 희소성과 긴급성은 디지털 제품부터 실제 상품과 서비스까지 모든 것을 판매하는 데 사용되어 왔으며, 전환을 유도하고 매출을 높이는 데 매우 효과적이라는 것이 입증되었습니다.\r\n\r\n희소성과 긴박감은 즉각적인 만족과 놓칠까 봐 두려워하는 인간의 본능적인 욕구(FOMO)를 활용하기 때문입니다. 고객이 빨리 행동하지 않으면 소중한 기회를 놓칠 수 있다고 생각하면 행동을 취하고 구매할 가능성이 높아집니다.\r\n\r\n또한 희소성과 긴박감은 사회적 증거 및 권위와 같은 다른 정신적 트리거와 결합하여 더욱 강력한 마케팅 캠페인을 만들 수 있습니다. 제품이나 서비스의 인기와 효과를 입증함으로써 기업은 프로모션의 가치와 긴급성을 높여 더 많은 판매와 수익으로 이어질 수 있습니다.\r\n\r\n전반적으로 희소성과 긴급성은 마케팅 캠페인에서 결과를 이끌어내는 것으로 입증된 강력한 정신적 트리거입니다. 이러한 트리거를 활용하여 기업은 제품이나 서비스에 대한 긴박감과 독점감을 조성하여 고객이 너무 늦기 전에 행동을 취하고 구매하도록 동기를 부여할 수 있습니다."
            },
            {
                "from": "human",
                "value": "러셀 브런슨의 완벽한 웨비나 스크립트 요약..."
            },
            {
                "from": "gpt",
                "value": "우리를 판매 Brunson의 완벽한 웨비나 스크립트는 제품 또는 서비스를 판매하도록 설계된 웨비나를 제공하기 위한 프레임워크입니다. 완벽한 웨비나 스크립트의 주요 구성 요소는 다음과 같습니다:\r\n\r\n1. 소개: 웹 세미나는 발표자의 신뢰성을 확립하고 주제를 설정하는 소개로 시작됩니다.\r\n2. 스토리: 그런 다음 발표자는 주제와 관련된 개인적인 이야기를 들려주며 청중과의 유대감을 형성합니다.\r\n3. 콘텐츠: 발표자는 주제와 관련된 가치 있는 콘텐츠를 제공하여 자신의 전문성을 입증하고 청중과 신뢰를 구축합니다.\r\n4. 제안: 발표자는 판매 중인 제품 또는 서비스를 소개하며 청중의 문제를 어떻게 해결하는지 설명하고 자신의 주장을 뒷받침할 수 있는 사회적 증거를 제공합니다.\r\n5. 보너스: 발표자는 보너스를 제공하여 거래를 더 달콤하게 만들고 제안을 더욱 매력적으로 만듭니다.\r\n6. 희소성: 발표자는 오퍼의 가용성을 제한하거나 구매 기한을 포함하여 긴박감을 조성합니다.\r\n7. 자주 묻는 질문: 발표자는 오퍼에 대한 일반적인 질문에 답변하여 이의 제기를 해결하고 청중과의 신뢰를 구축합니다.\r\n8. 마무리: 그런 다음 발표자는 명확한 클릭 유도 문안을 전달하여 청중이 제품 또는 서비스를 구매하기 위해 특정 조치를 취하도록 안내합니다.\r\n\r\n전반적으로 완벽한 웹 세미나 스크립트는 청중과의 신뢰를 구축하고 발표자의 신뢰성을 확립하며 가치 있는 콘텐츠를 제공하는 동시에 제시되는 제안에 대한 긴박감과 희소성을 조성하도록 설계되었습니다. 이 프레임워크를 따르면 기업은 웨비나를 타겟 고객에게 제품과 서비스를 판매하는 강력한 도구로 사용할 수 있습니다."
            },
            {
                "from": "human",
                "value": "토니 로빈스가 설명하는 인간의 6가지 욕구를 요약해 보세요..."
            },
            {
                "from": "gpt",
                "value": "토니 로빈스는 인간의 행동과 동기를 이끄는 6가지 핵심 욕구를 설명합니다. 이 여섯 가지 욕구는 다음과 같습니다:\r\n\r\n1. 확실성: 안전, 안정성, 예측 가능성에 대한 욕구. 여기에는 편안함, 보안, 환경에 대한 통제에 대한 욕구가 포함됩니다.\r\n2. 다양성: 참신함, 흥분, 변화에 대한 욕구. 여기에는 모험, 자극, 새로운 경험에 대한 욕구가 포함됩니다.\r\n3. 중요성: 중요성: 중요하고 특별하며 독특하다고 느끼고자 하는 욕구. 여기에는 인정, 성취, 존경에 대한 욕구가 포함됩니다.\r\n4. 연결: 사회적 연결과 사랑에 대한 욕구. 여기에는 친밀감, 소속감, 우정에 대한 욕구가 포함됩니다.\r\n5. 성장: 성장: 개인적인 성장과 발전의 욕구. 여기에는 학습, 자기계발, 진보에 대한 욕구가 포함됩니다.\r\n6. 기여: 세상에 긍정적인 영향을 미치고 싶은 욕구. 여기에는 환원하고, 다른 사람을 돕고, 변화를 만들고자 하는 욕구가 포함됩니다.\r\n\r\n토니 로빈스에 따르면 이 여섯 가지 욕구는 보편적인 것으로 모든 개인에게 적용됩니다. 사람마다 이러한 욕구의 우선순위는 다를 수 있고 충족하는 방식도 다를 수 있지만, 이러한 욕구는 인간 행동의 근본적인 동인입니다. 이러한 욕구를 이해함으로써 개인은 자신의 동기와 행동에 대한 통찰력을 얻을 수 있으며, 이러한 지식을 활용하여 보다 만족스럽고 의미 있는 삶을 만들 수 있습니다."
            }
            ]
        },
    ]
    print(SupervisedDataset(raw_data, tokenizer=tokenizer))


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
