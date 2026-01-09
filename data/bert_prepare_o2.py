from __future__ import annotations
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

TRAIN_FILE = "train.jsonl"
VAL_FILE   = "val.jsonl"
TEST_FILE  = "test.jsonl"

OUT_DIR    = Path("../data/tokenized/o2") # Nota: nuova directory

MODEL_ID   = "answerdotai/ModernBERT-base"
MAX_LEN    = 128
IGNORE_IDX = -100

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
        "test": TEST_FILE
    }
    
    print(f"Loading data files: {data_files}")
    ds = load_dataset("json", data_files=data_files)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(example):
        t0 = example["task"]
        output = {
            "task": t0,
            "id": example.get("id", ""),
            "label_a": IGNORE_IDX,
            "label_b": IGNORE_IDX,
            # Placeholder per task A
            "input_ids": [], "attention_mask": [],
            # Placeholder per task B
            "input_ids_sarc": [], "attention_mask_sarc": [],
            "input_ids_reph": [], "attention_mask_reph": [],
        }

        if t0 == "A":
            txt = example.get("text") or ""
            tok = tokenizer(txt, truncation=True, max_length=MAX_LEN)
            output["input_ids"] = tok["input_ids"]
            output["attention_mask"] = tok["attention_mask"]
            
            if example.get("label_a") is not None:
                output["label_a"] = int(example["label_a"])

        elif t0 == "B":
            # Task B: Tokenizziamo SEPARATAMENTE le due frasi
            txt_sarc = example.get("text_a") or "" # text_a è la frase sarcastica nel dataset
            txt_reph = example.get("text_b") or "" # text_b è la rephrase
            
            # 1. Frase Sarcastica
            tok_sarc = tokenizer(txt_sarc, truncation=True, max_length=MAX_LEN)
            output["input_ids_sarc"] = tok_sarc["input_ids"]
            output["attention_mask_sarc"] = tok_sarc["attention_mask"]

            # 2. Frase Rephrase
            tok_reph = tokenizer(txt_reph, truncation=True, max_length=MAX_LEN)
            output["input_ids_reph"] = tok_reph["input_ids"]
            output["attention_mask_reph"] = tok_reph["attention_mask"]
            
            # Nota: Label B non serve più realmente per il training ranking, 
            # ma lo teniamo per compatibilità. Sappiamo che text_a > text_b.
            if example.get("label_b") is not None:
                output["label_b"] = int(example["label_b"])
        
        return output

    print("Tokenizing dataset for Siamese/Ranking...")
    raw_cols = ds["train"].column_names
    ds_tok = ds.map(tokenize, batched=False, remove_columns=raw_cols)

    print(f"Columns: {ds_tok['train'].column_names}")
    ds_tok.save_to_disk(str(OUT_DIR))
    print(f"Saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()