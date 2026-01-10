from __future__ import annotations
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

DIR = Path("o1_multitask")
TRAIN_FILE = DIR / "train.jsonl"
VAL_FILE   = DIR / "val.jsonl"
TEST_FILE  = DIR / "test.jsonl"

OUT_DIR    = Path("tokenized/o1")

MODEL_ID   = "answerdotai/ModernBERT-base"
MAX_LEN_A  = 128
MAX_LEN_B  = 128
IGNORE_IDX = -100

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    }
    if TEST_FILE.exists():
        data_files["test"] = str(TEST_FILE)

    print(f"Loading data files: {data_files}")
    ds = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(example):
        t0 = example["task"]

        l_a = IGNORE_IDX
        l_b = IGNORE_IDX

        if t0 == "A":
            txt = example.get("text") or ""
            tok = tokenizer(
                txt,
                truncation=True,
                max_length=MAX_LEN_A,
                add_special_tokens=True,
            )
            val_a = example.get("label_a")
            if val_a is not None:
                l_a = int(val_a)

        elif t0 == "B":
            txt_a = example.get("text_a") or ""
            txt_b = example.get("text_b") or ""
            tok = tokenizer(
                txt_a,
                txt_b,
                truncation=True,
                max_length=MAX_LEN_B,
                add_special_tokens=True,
            )
            val_b = example.get("label_b")
            if val_b is not None:
                l_b = int(val_b)
        else:
            raise ValueError(f"Unknown task='{t0}' in id={example.get('id')}")

        tok["label_a"] = l_a
        tok["label_b"] = l_b
        tok["task"] = t0
        tok["id"] = example.get("id", "")
        return tok

    raw_cols = ds["train"].column_names
    print("Tokenizing dataset...")

    ds_tok = ds.map(tokenize, batched=False, remove_columns=raw_cols)

    print(f"Columns in train: {ds_tok['train'].column_names}")
    ds_tok.save_to_disk(str(OUT_DIR))
    print(f"Saved tokenized dataset to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()