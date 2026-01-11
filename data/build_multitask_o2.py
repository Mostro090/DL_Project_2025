import json
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TRAIN_FILE = "isarcasmeval_train.jsonl"
TEST_FILE  = "isarcasmeval_test.jsonl"
OUTPUT_DIR = Path("o2_multitask")

def create_multitask_rows(df: pd.DataFrame, include_B: bool) -> list[dict]:
    rows: list[dict] = []

    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        label = int(row["label"])
        row_id = str(row.get("id", idx))

        rows.append({
            "task": "A",
            "id": row_id,
            "text": text,
            "label_a": label
        })

        if include_B:
            rephrase = row.get("rephrase", None)
            if label == 1 and isinstance(rephrase, str) and rephrase.strip():
                r = rephrase.strip()
                
                rows.append({
                    "task": "B", 
                    "id": row_id, 
                    "text_a": text, 
                    "text_b": r     
                })

    return rows

def save_jsonl(rows: list[dict], filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def main() -> None:
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Missing {TRAIN_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_full = read_jsonl(TRAIN_FILE)
    
    train_df, val_df = train_test_split(
        df_full, test_size=0.1, random_state=42, stratify=df_full["label"]
    )

    print(f"Generating O2 (Canonical Single Rows) into {OUTPUT_DIR}...")

    save_jsonl(create_multitask_rows(train_df, include_B=True), "train.jsonl")
    
    save_jsonl(create_multitask_rows(val_df, include_B=False), "val.jsonl")

    if os.path.exists(TEST_FILE):
        df_test = read_jsonl(TEST_FILE)
        save_jsonl(create_multitask_rows(df_test, include_B=False), "test.jsonl")

    d = pd.read_json(OUTPUT_DIR / "train.jsonl", lines=True)
    print("Distribuzione Task nel Train:")
    print(d["task"].value_counts())

if __name__ == "__main__":
    main()