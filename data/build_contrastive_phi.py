import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def build_task_A_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in rows:
        out.append({
            "task": "A",
            "text": ex["text"],
            "label": int(ex["label"]),
        })
    return out


def build_task_D_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in rows:
        label = int(ex["label"])
        rephrase = ex.get("rephrase", None)
        if label == 1 and rephrase:
            out.append({
                "task": "D",
                "text": ex["text"],        
                "label": 1,
                "rephrase": rephrase,     
            })
    return out


def count_tasks(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        t = r.get("task", "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return counts


def build_multitask_jsonl(
    train_jsonl: str,
    test_jsonl: str,
    out_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    include_D_in_val: bool = True,
) -> None:
    train_rows_all = load_jsonl(train_jsonl)
    test_rows = load_jsonl(test_jsonl)

    train_rows, val_rows = stratified_split_train_val(train_rows_all, val_ratio, seed)

    train_out = build_task_A_rows(train_rows) + build_task_D_rows(train_rows)
    val_out = build_task_A_rows(val_rows) + (build_task_D_rows(val_rows) if include_D_in_val else [])
    test_out = build_task_A_rows(test_rows)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    save_jsonl(str(out_path / "train.jsonl"), train_out)
    save_jsonl(str(out_path / "validation.jsonl"), val_out)
    save_jsonl(str(out_path / "test.jsonl"), test_out)

    print("Saved JSONL to:", out_path.resolve())
    print("Train counts:", count_tasks(train_out))
    print("Val counts:", count_tasks(val_out))
    print("Test counts:", count_tasks(test_out))


def main() -> None:
    train_jsonl = "isarcasmeval_train.jsonl"
    test_jsonl = "isarcasmeval_test.jsonl"
    out_dir = "phi3_contrastive"
    val_ratio = 0.1
    seed = 42

    build_multitask_jsonl(
        train_jsonl=train_jsonl,
        test_jsonl=test_jsonl,
        out_dir=out_dir,
        val_ratio=val_ratio,
        seed=seed,
        include_D_in_val=True,
    )


if __name__ == "__main__":
    main()
