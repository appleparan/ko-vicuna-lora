import json
import os
import sys
from pathlib import Path
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset


def main():
    base_dir = (Path(__file__) / Path("../../../data/dataset")).resolve()
    base_dir = Path("./data").resolve()

    raw_data_files = {}
    qa_data_files = {}
    for case in ["train", "valid", "test"]:
        files = list(sorted((Path("./data") / case).glob("*.json")))
        raw_data_files[case] = str(
            (base_dir / f"bioleaflets_{case}_raw.jsonl").resolve()
        )
        qa_data_files[case] = str(
            (base_dir / f"bioleaflets_{case}_qa.jsonl").resolve()
        )

        f_raw = open(base_dir / f"bioleaflets_{case}_raw.jsonl", "w")
        f_qa = open(base_dir / f"bioleaflets_{case}_qa.jsonl", "w")
        for file_name in files:
            f = open(file_name, "r")
            j_qa = {}
            j_raw = json.load(f)
            j_qa["ID"] = j_raw["ID"]
            for i in range(6):
                j_qa[f"Section_{i + 1}"] = {}
                j_qa[f"Section_{i + 1}"]["question"] = j_raw[
                    f"Section_{i + 1}"
                ]["Title"]
                j_qa[f"Section_{i + 1}"]["answer"] = j_raw[f"Section_{i + 1}"][
                    "Section_Content"
                ]

                del j_raw[f"Section_{i + 1}"]
            f_raw.write(json.dumps(j_raw) + "\n")
            f_qa.write(json.dumps(j_qa) + "\n")
            f.close()
        f_raw.close()
        f_qa.close()

    raw_dataset = load_dataset("json", data_files=raw_data_files)
    qa_dataset = load_dataset("json", data_files=qa_data_files)
    print(raw_dataset)
    print(qa_dataset)


if __name__ == "__main__":
    main()
