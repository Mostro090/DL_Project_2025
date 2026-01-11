from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

def iter_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue

def find_candidate_files(folder: Path) -> list[Path]:
    exts = {".jsonl", ".json", ".txt"}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(val))
    
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_row = "| " + " | ".join("-" * w for w in col_widths) + " |"
    data_rows = []
    for r in rows:
        data_rows.append("| " + " | ".join(val.ljust(w) for val, w in zip(r, col_widths)) + " |")
    
    return "\n".join([header_row, sep_row] + data_rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="o2_multitask")
    parser.add_argument("--output", default="report_o2_quantitative.md")
    parser.add_argument("--pos-label", type=int, default=1)
    args = parser.parse_args()

    base = Path(args.input).resolve()
    if not base.exists():
        raise SystemExit(f"Path not found: {base}")

    files = find_candidate_files(base)
    
    stats = {
        "total_rows": 0,
        "A_total": 0,
        "A_pos": 0,
        "A_neg": 0,
        "B_boost": 0
    }

    for fp in files:
        for obj in iter_json_lines(fp):
            stats["total_rows"] += 1
            task = obj.get("task")
            
            if task == "A":
                stats["A_total"] += 1
                lbl = obj.get("label_a")
                if lbl == args.pos_label:
                    stats["A_pos"] += 1
                else:
                    stats["A_neg"] += 1 
            elif task == "B":
                stats["B_boost"] += 1

    a_pos = stats["A_pos"]
    a_neg = stats["A_neg"]
    b_boost = stats["B_boost"]
    
    total_positive_signal = a_pos + b_boost
    ratio_raw = a_neg / a_pos if a_pos > 0 else 0
    ratio_boosted = a_neg / total_positive_signal if total_positive_signal > 0 else 0
    boost_pct = (b_boost / a_pos * 100) if a_pos > 0 else 0.0
    improvement_pct = ((ratio_raw - ratio_boosted) / ratio_raw * 100) if ratio_raw > 0 else 0

    md = []
    md.append(f"# Report Dataset: Boosting Strategy")
    md.append(f"**Dataset:** `{base.name}` | **Rows:** {stats['total_rows']}\n")

    md.append("## 1. Initial State (Task A Only)")
    
    table_a = [
        ["Class", "Count", "Percentage"],
        ["Negative (0)", f"{a_neg}", f"{(a_neg/stats['A_total']*100):.1f}%"],
        ["Positive (1)", f"{a_pos}", f"{(a_pos/stats['A_total']*100):.1f}%"],
    ]
    md.append(md_table(table_a[0], table_a[1:]))
    md.append(f"\n> **Initial Ratio:** 1 Positive per **{ratio_raw:.1f}** Negatives.\n")

    md.append("## 2. Boosting Mechanism (Task B)")
    
    table_b = [
        ["Signal Source", "Volume"],
        ["Task A (Positive)", str(a_pos)],
        ["Task B (Boost)", str(b_boost)],
        ["**Total Positive Signal**", f"**{total_positive_signal}**"]
    ]
    md.append(md_table(table_b[0], table_b[1:]))
    
    md.append("\n## 3. Balancing Effectiveness")
    
    table_impact = [
        ["Scenario", "Neg vs Pos", "Ratio (Neg:Pos)"],
        ["Classification Only", f"{a_neg} vs {a_pos}", f"{ratio_raw:.2f} : 1"],
        ["Multitask (Boosted)", f"{a_neg} vs {total_positive_signal}", f"**{ratio_boosted:.2f} : 1**"]
    ]
    md.append(md_table(table_impact[0], table_impact[1:]))
    
    md.append("\n### Quantitative Conclusion")
    md.append(f"- Positive information volume increase: **+{boost_pct:.1f}%**.")
    md.append(f"- Effective imbalance reduction: **{improvement_pct:.1f}%** (from {ratio_raw:.1f}:1 to {ratio_boosted:.1f}:1).")

    out_path = Path(args.output).resolve()
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Report generated: {out_path}")

if __name__ == "__main__":
    main()