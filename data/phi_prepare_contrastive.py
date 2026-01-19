import json
from pathlib import Path
from typing import List, Dict, Any

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

    enc_full = tok(full_text, truncation=True, max_length=max_len, add_special_tokens=False)
    enc_prompt = tok(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(enc_prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_prompt_only(
    tok: AutoTokenizer,
    prompt: str,
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
    else:
        prompt_text = prompt

    enc = tok(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def count_tasks(ds: Dataset) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in ds["task"]:
        counts[t] = counts.get(t, 0) + 1
    return counts


def prepare_from_multitask_jsonl(
    jsonl_dir: str,
    model_name: str,
    out_dir: str,
    max_len: int = 1024,
    use_chat_template: bool = True,
) -> None:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eos = tok.eos_token or ""

    jsonl_path = Path(jsonl_dir)
    train_rows = load_jsonl(str(jsonl_path / "train.jsonl"))
    val_rows = load_jsonl(str(jsonl_path / "validation.jsonl"))
    test_rows = load_jsonl(str(jsonl_path / "test.jsonl"))

    def build_tokenized(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ex in rows:
            task = ex["task"]

            if task == "A":
                text = ex["text"]
                label = int(ex["label"])
                prompt = build_prompt_A(text)
                target = build_target_A(label, eos=eos)
                toks = tokenize_with_masking(tok, prompt, target, max_len, use_chat_template)
                toks["task"] = "A"
                toks["gold_label"] = label
                out.append(toks)

            elif task == "D":
                label = int(ex.get("label", 0))
                rephrase = ex.get("rephrase", None)
                if label == 1 and rephrase:
                    text_s = ex["text"]
                    text_r = rephrase

                    prompt_s = build_prompt_A(text_s)
                    prompt_r = build_prompt_A(text_r)

                    toks_s = tokenize_prompt_only(tok, prompt_s, max_len, use_chat_template)
                    toks_r = tokenize_prompt_only(tok, prompt_r, max_len, use_chat_template)

                    out.append({
                        "task": "D",
                        "input_ids_s": toks_s["input_ids"],
                        "attention_mask_s": toks_s["attention_mask"],
                        "input_ids_r": toks_r["input_ids"],
                        "attention_mask_r": toks_r["attention_mask"],
                    })

            else:
                raise ValueError(f"Unknown task: {task}")

        return out

    ds_train = Dataset.from_list(build_tokenized(train_rows))
    ds_val = Dataset.from_list(build_tokenized(val_rows))
    ds_test = Dataset.from_list(build_tokenized(test_rows))

    ds = DatasetDict(train=ds_train, validation=ds_val, test=ds_test)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    print(ds)
    print("Train counts:", count_tasks(ds_train))
    print("Val counts:", count_tasks(ds_val))
    print("Test counts:", count_tasks(ds_test))
    print("Saved to:", out_path.resolve())


def main() -> None:
    jsonl_dir = "phi3_contrastive"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    out_dir = "phi3_dataset_contrastive"
    max_len = 1024
    use_chat_template = True

    prepare_from_multitask_jsonl(
        jsonl_dir=jsonl_dir,
        model_name=model_name,
        out_dir=out_dir,
        max_len=max_len,
        use_chat_template=use_chat_template,
    )


if __name__ == "__main__":
    main()