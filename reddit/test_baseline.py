import os
import re
import json
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from groq import AsyncGroq  

INPUT_JSONL = Path("reddit_sample_20k.jsonl")

KIMI_MODEL = "moonshotai/kimi-k2-instruct-0905"
LLAMA_MODEL = "llama-3.3-70b-versatile"

REPORT_MD = Path(f"report_llm.md")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", ":)") 

MAX_CONCURRENCY = 15
MAX_RETRIES = 3

SYSTEM_MSG = (
    "You are a strict binary classifier for sarcasm detection.\n"
    "Given a short social media post, decide if it is sarcastic.\n"
    "Output ONLY a single character:\n"
    "1 = sarcastic\n"
    "0 = not sarcastic"
)

_LABEL_RE = re.compile(r"^\s*([01])\s*$")

def parse_label(text: Optional[str]) -> Optional[int]:
    """Accetta SOLO '0' o '1' (ignorando spazi)."""
    if not text:
        return None
    m = _LABEL_RE.match(text)
    if not m:
        return None
    return int(m.group(1))

def safe_int_label(x: Any) -> Optional[int]:
    try:
        v = int(x)
        return v if v in (0, 1) else None
    except Exception:
        return None

@dataclass
class BinaryMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> BinaryMetrics:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return BinaryMetrics(tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def simple_classification_report(y_true: List[int], y_pred: List[int]) -> str:
    m = compute_binary_metrics(y_true, y_pred)
    precision_0 = m.tn / (m.tn + m.fn) if (m.tn + m.fn) else 0.0
    recall_0 = m.tn / (m.tn + m.fp) if (m.tn + m.fp) else 0.0
    f1_0 = (2 * precision_0 * recall_0 / (precision_0 + recall_0)) if (precision_0 + recall_0) else 0.0
    support_0 = m.tn + m.fp

    precision_1 = m.precision
    recall_1 = m.recall
    f1_1 = m.f1
    support_1 = m.tp + m.fn

    total = support_0 + support_1
    macro_p = (precision_0 + precision_1) / 2
    macro_r = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total if total else 0.0

    lines = []
    lines.append("              precision    recall  f1-score   support")
    lines.append(f"           0     {precision_0:0.4f}    {recall_0:0.4f}    {f1_0:0.4f}      {support_0}")
    lines.append(f"           1     {precision_1:0.4f}    {recall_1:0.4f}    {f1_1:0.4f}      {support_1}")
    lines.append("")
    lines.append(f"    accuracy                         {m.accuracy:0.4f}      {total}")
    lines.append(f"   macro avg     {macro_p:0.4f}    {macro_r:0.4f}    {macro_f1:0.4f}      {total}")
    lines.append(f"weighted avg                     {weighted_f1:0.4f}      {total}")
    return "\n".join(lines)

def build_markdown_report(title: str, report_text: str) -> str:
    md: List[str] = []
    md.append(f"# {title}\n")
    md.append("## Report Dettagliato\n")
    md.append("```text")
    md.append(report_text)
    md.append("```")
    return "\n".join(md)

async def classify_with_model(
    client: AsyncGroq,
    model: str,
    txt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": f"Text:\n{txt}\n\nAnswer (0 or 1):"},
                    ],
                    temperature=0.0,
                    max_completion_tokens=5,
                )
                raw = resp.choices[0].message.content
                pred = parse_label(raw)
                if pred is not None:
                    return {"pred": pred, "raw": raw, "error": None}
                return {"pred": None, "raw": raw, "error": "parsing_failed"}

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {"pred": None, "raw": None, "error": str(e)}
                await asyncio.sleep(1 * (2 ** attempt))

    return {"pred": None, "raw": None, "error": "unknown"}

async def process_single_item(
    client: AsyncGroq,
    semaphore: asyncio.Semaphore,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    txt = item.get("text", "")

    kimi = await classify_with_model(client, KIMI_MODEL, txt, semaphore)
    llama = await classify_with_model(client, LLAMA_MODEL, txt, semaphore)

    out = dict(item)
    out["kimi_pred"] = kimi["pred"]
    out["kimi_raw"] = kimi["raw"]
    if kimi["error"]:
        out["kimi_error"] = kimi["error"]

    out["llama_pred"] = llama["pred"]
    out["llama_raw"] = llama["raw"]
    if llama["error"]:
        out["llama_error"] = llama["error"]

    return out

async def main():
    if not INPUT_JSONL.exists():
        print(f"File non trovato: {INPUT_JSONL}")
        return

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Caricati {len(data)} esempi. Valutazione con Kimi + Llama...")

    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [process_single_item(client, semaphore, item) for item in data]

    results: List[Dict[str, Any]] = []
    total = len(tasks)

    for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
        res = await fut
        results.append(res)
        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] completati...")
    
    print("Elaborazione completata. Calcolo metriche...")

    y_true_kimi: List[int] = []
    y_pred_kimi: List[int] = []

    y_true_llama: List[int] = []
    y_pred_llama: List[int] = []

    for r in results:
        y = safe_int_label(r.get("label"))
        if y is None:
            continue

        kp = r.get("kimi_pred")
        if kp in (0, 1):
            y_true_kimi.append(y)
            y_pred_kimi.append(int(kp))
        
        lp = r.get("llama_pred")
        if lp in (0, 1):
            y_true_llama.append(y)
            y_pred_llama.append(int(lp))

    md_parts: List[str] = []
    
    if not y_true_kimi and not y_true_llama:
        print("Nessuna predizione valida trovata.")
        return

    if y_true_kimi:
        report_text_kimi = simple_classification_report(y_true_kimi, y_pred_kimi)
        md_parts.append(
            build_markdown_report(
                title="Report Valutazione Sarcasmo (Kimi)",
                report_text=report_text_kimi
            )
        )
        md_parts.append("\n---\n")

    if y_true_llama:
        report_text_llama = simple_classification_report(y_true_llama, y_pred_llama)
        md_parts.append(
            build_markdown_report(
                title="Report Valutazione Sarcasmo (Llama)",
                report_text=report_text_llama
            )
        )

    REPORT_MD.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"Report salvato in: {REPORT_MD.as_posix()}")

if __name__ == "__main__":
    asyncio.run(main())