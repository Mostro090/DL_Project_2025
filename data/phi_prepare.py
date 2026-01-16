import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

IGNORE_INDEX = -100

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_prompt_A(text: str) -> str:
    return (
        "Classify the following text.\n"
        'Reply ONLY with "A" if it is sarcastic, or "B" if it is NOT sarcastic.\n'
        f"Text: {text}\n"
        "Answer: "
    )

def build_target_A(label: int, eos: str) -> str:
    return ("A" if int(label) == 1 else "B") + eos

def build_prompt_C(text: str) -> str:
    return (
        "Rewrite the text into a literal, non-sarcastic version while preserving the meaning.\n"
        f"Text: {text}\n"
        "Rewrite: "
    )

def build_target_C(rephrase: str, eos: str) -> str:
    return rephrase + eos

def tokenize_with_masking(
    tok: AutoTokenizer,
    prompt: str,
    target: str,
    max_len: int,
    use_chat_template: bool,
    system_msg: str = "You are a helpful assistant."
) -> Dict[str, Any]:

    if use_chat_template:
        messages_prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tok.apply_chat_template(
            messages_prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        full_text = prompt_text + target
    else:
        full_text = prompt + target
        prompt_text = prompt

    enc_full = tok(
        full_text, truncation=True, max_length=max_len, add_special_tokens=False
    )
    enc_prompt = tok(
        prompt_text, truncation=True, max_length=max_len, add_special_tokens=False
    )

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = len(enc_prompt["input_ids"])
    
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def build_task_A_examples(
    rows: List[Dict[str, Any]],
    tok: AutoTokenizer,
    max_len: int,
    use_chat_template: bool
) -> List[Dict[str, Any]]:
    eos = tok.eos_token or ""
    out = []
    for ex in rows:
        text = ex["text"]
        label = int(ex["label"])

        prompt = build_prompt_A(text)
        target = build_target_A(label, eos=eos)
        toks = tokenize_with_masking(tok, prompt, target, max_len, use_chat_template)

        toks["task"] = "A"
        toks["gold_label"] = label
        out.append(toks)
    return out

def build_task_C_examples(
    rows: List[Dict[str, Any]],
    tok: AutoTokenizer,
    max_len: int,
    use_chat_template: bool
) -> List[Dict[str, Any]]:
    eos = tok.eos_token or ""
    out = []
    for ex in rows:
        label = int(ex["label"])
        rephrase = ex.get("rephrase", None)

        if label == 1 and rephrase:
            text = ex["text"]
            prompt = build_prompt_C(text)
            target = build_target_C(rephrase, eos=eos)
            toks = tokenize_with_masking(tok, prompt, target, max_len, use_chat_template)

            toks["task"] = "C"
            toks["gold_label"] = 1
            out.append(toks)
    return out

def stratified_split_train_val(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)

    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_val = int(round(len(pos) * val_ratio))
    n_neg_val = int(round(len(neg) * val_ratio))

    val_rows = pos[:n_pos_val] + neg[:n_neg_val]
    train_rows = pos[n_pos_val:] + neg[n_neg_val:]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows

def prepare_datasets(
    train_jsonl: str,
    test_jsonl: str,
    model_name: str,
    out_dir: str,
    val_ratio: float = 0.1,
    max_len: int = 1024,
    use_chat_template: bool = True,
    seed: int = 42
) -> None:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows_all = load_jsonl(train_jsonl)
    test_rows = load_jsonl(test_jsonl)

    train_rows, val_rows = stratified_split_train_val(train_rows_all, val_ratio, seed)

    train_A = build_task_A_examples(train_rows, tok, max_len, use_chat_template)
    train_C = build_task_C_examples(train_rows, tok, max_len, use_chat_template)

    val_A = build_task_A_examples(val_rows, tok, max_len, use_chat_template)
    val_C = build_task_C_examples(val_rows, tok, max_len, use_chat_template)

    test_A = build_task_A_examples(test_rows, tok, max_len, use_chat_template)

    ds_train = Dataset.from_list(train_A + train_C)
    ds_val = Dataset.from_list(val_A + val_C)
    ds_test = Dataset.from_list(test_A)

    ds = DatasetDict(train=ds_train, validation=ds_val, test=ds_test)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    print(ds)
    print("Train counts:", count_tasks(ds_train))
    print("Val counts:", count_tasks(ds_val))
    print("Test counts:", count_tasks(ds_test))
    print("Saved to:", out_path.resolve())

def count_tasks(ds: Dataset) -> Dict[str, int]:
    counts = {}
    for t in ds["task"]:
        counts[t] = counts.get(t, 0) + 1
    return counts

if __name__ == "__main__":
    prepare_datasets(
        train_jsonl="isarcasmeval_train.jsonl",
        test_jsonl="isarcasmeval_test.jsonl",
        model_name="microsoft/Phi-3-mini-4k-instruct",
        out_dir="phi3_dataset",
        val_ratio=0.1,
        max_len=1024,
        use_chat_template=True,
        seed=42
    )