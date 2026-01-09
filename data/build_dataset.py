import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class SarcasmExample:
    text: str
    label: Optional[int]
    rephrase: Optional[str] = None

def _to_optional_str(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _to_optional_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


def _pick_first_existing(row: pd.Series, candidates: List[str]) -> Optional[Any]:
    for c in candidates:
        if c in row and not pd.isna(row[c]):
            return row[c]
    return None


def download_and_process_isarcasmeval(
    split: str,
    save_path: str,
    dataset_urls: Optional[Dict[str, str]] = None,
) -> List[SarcasmExample]:
    default_urls = {
        "train": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
        "test": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv",
    }
    urls = dataset_urls or default_urls

    if split not in urls:
        raise ValueError(f"split must be one of {list(urls.keys())}, got: {split}")

    url = urls[split]
    print(f"Downloading {split} set from: {url} ...")

    try:
        df = pd.read_csv(url)
        examples: List[SarcasmExample] = []

        print(f"Processing {len(df)} records...")

        for _, row in df.iterrows():
            text_val = _pick_first_existing(row, ["tweet", "text", "sentence", "content"])
            if text_val is None:
                continue

            label_val = None
            if "sarcastic" in df.columns:
                label_val = _to_optional_int(row.get("sarcastic"))
            elif "label" in df.columns:
                label_val = _to_optional_int(row.get("label"))

            rephrase_val = None
            if "rephrase" in df.columns:
                rephrase_val = _to_optional_str(row.get("rephrase"))

            examples.append(
                SarcasmExample(
                    text=str(text_val).strip(),
                    label=label_val,
                    rephrase=rephrase_val,
                )
            )

        print(f"Saving {len(examples)} records to '{save_path}'...")
        with open(save_path, "w", encoding="utf-8") as f:
            for item in examples:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

        labeled = [x for x in examples if x.label is not None]
        if labeled:
            sarcastic_count = sum(1 for x in labeled if x.label == 1)
            non_sarcastic_count = sum(1 for x in labeled if x.label == 0)
            print(
                f"\nStatistics ({split}):\n"
                f"Total: {len(examples)}\n"
                f"Labeled: {len(labeled)}\n"
                f"Sarcastic: {sarcastic_count}\n"
                f"Non Sarcastic: {non_sarcastic_count}"
            )
        else:
            print(f"\nStatistics ({split}):\nTotal: {len(examples)}\n(No labels found)")

        print("Completed successfully.")
        return examples

    except Exception as e:
        print(f"Error during process ({split}): {e}")
        return []


if __name__ == "__main__":
    train_data = download_and_process_isarcasmeval(split="train", save_path="isarcasmeval_train.jsonl")
    test_data = download_and_process_isarcasmeval(split="test", save_path="isarcasmeval_test.jsonl")

    if train_data:
        print("\n--- Train Preview (First 2 records) ---")
        for ex in train_data[:2]:
            print(ex)

    if test_data:
        print("\n--- Test Preview (First 2 records) ---")
        for ex in test_data[:2]:
            print(ex)
