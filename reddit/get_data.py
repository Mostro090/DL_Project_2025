from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

DATASET_NAME = "marcbishara/sarcasm-on-reddit"
SPLIT = "holdout"
N_SAMPLES = 20000
SEED = 42
BUFFER_SIZE = 100000
OUT_PATH = Path("./reddit_sample_20k.jsonl")

DELETED = {"[deleted]", "[removed]"}

def norm_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    return x.strip()

def is_valid_comment(x: str) -> bool:
    return bool(x) and (x.lower() not in DELETED)

def label_to_int(label: Any) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return 1 if label == 1 else 0
    if isinstance(label, str):
        l = label.strip().lower()
        if l == "sarcastic":
            return 1
        if l == "not_sarcastic":
            return 0
    raise ValueError(f"Unknown label format: {label} ({type(label)})")

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)

    kept = 0
    dropped = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in ds:
            if kept >= N_SAMPLES:
                break

            text = norm_text(ex.get("comment"))
            if not is_valid_comment(text):
                dropped += 1
                continue

            try:
                y = label_to_int(ex.get("label"))
            except Exception:
                dropped += 1
                continue

            f.write(json.dumps({"text": text, "label": y}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Saved: {OUT_PATH.resolve()} | kept={kept} dropped={dropped}")

if __name__ == "__main__":
    main()