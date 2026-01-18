from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

ADAPTER_DIR = None # se vuoi testare il fine - tuned metti "../results/runs/phi3_o1/step_400"
REDDIT_JSONL = Path("./reddit_sample_20k.jsonl")  
MODEL_ID_FALLBACK = "microsoft/Phi-3-mini-4k-instruct"

BATCH_SIZE = 8
MAX_PROMPT_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.0

REPORT_OUT = Path("report_phi3.md")
SCORES_OUT = Path("reddit_scores_phi3.jsonl")

def build_prompt_A(text: str) -> str:
    return (
        "Classify the following text.\n"
        'Reply ONLY with "A" if it is sarcastic, or "B" if it is NOT sarcastic.\n'
        f"Text: {text}\n"
        "Answer: "
    )

@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int
    support_0: int
    support_1: int


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    support_0 = int(np.sum(y_true == 0))
    support_1 = int(np.sum(y_true == 1))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

    return BinaryMetrics(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        tn=tn, fp=fp, fn=fn, tp=tp,
        support_0=support_0,
        support_1=support_1,
    )


def classification_report_like(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    m_pos = binary_metrics(y_true, y_pred)

    tp0, fp0, fn0 = m_pos.tn, m_pos.fn, m_pos.fp
    prec0 = tp0 / max(1, (tp0 + fp0))
    rec0 = tp0 / max(1, (tp0 + fn0))
    f10 = (2 * prec0 * rec0) / max(1e-12, (prec0 + rec0))

    prec1, rec1, f11 = m_pos.precision, m_pos.recall, m_pos.f1

    acc = m_pos.accuracy
    support0, support1 = m_pos.support_0, m_pos.support_1
    total = support0 + support1

    macro_prec = (prec0 + prec1) / 2
    macro_rec = (rec0 + rec1) / 2
    macro_f1 = (f10 + f11) / 2

    weighted_prec = (prec0 * support0 + prec1 * support1) / max(1, total)
    weighted_rec = (rec0 * support0 + rec1 * support1) / max(1, total)
    weighted_f1 = (f10 * support0 + f11 * support1) / max(1, total)

    lines: List[str] = []
    lines.append("                precision    recall  f1-score   support")
    lines.append("")
    lines.append(f"Non Sarcastic   {prec0:9.2f}  {rec0:8.2f}  {f10:8.2f}  {support0:8d}")
    lines.append(f"Sarcastic       {prec1:9.2f}  {rec1:8.2f}  {f11:8.2f}  {support1:8d}")
    lines.append("")
    lines.append(f"accuracy                            {acc:5.2f}  {total:8d}")
    lines.append(f"macro avg       {macro_prec:9.2f}  {macro_rec:8.2f}  {macro_f1:8.2f}  {total:8d}")
    lines.append(f"weighted avg    {weighted_prec:9.2f}  {weighted_rec:8.2f}  {weighted_f1:8.2f}  {total:8d}")
    return "\n".join(lines)

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_phi3_with_lora(adapter_path: Optional[Path]) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    if adapter_path is None or not Path(adapter_path).exists():
        print(f"Adapter non trovato o non specificato. Carico modello BASE: {MODEL_ID_FALLBACK}")
        tok = AutoTokenizer.from_pretrained(MODEL_ID_FALLBACK, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_FALLBACK,
            torch_dtype="auto",
            device_map="auto" if DEVICE.startswith("cuda") else None,
        )
        base_name = MODEL_ID_FALLBACK
    else:
        print(f"Trovato adapter LoRA: {adapter_path}")
        peft_cfg = PeftConfig.from_pretrained(str(adapter_path))
        base_name = peft_cfg.base_model_name_or_path or MODEL_ID_FALLBACK
        
        try:
            tok = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype="auto",
            device_map="auto" if DEVICE.startswith("cuda") else None,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model.eval()
    return model, tok, base_name

@torch.no_grad()
def score_batch_next_token_AB(
    model,
    tok,
    prompt_ids_batch: List[List[int]],
    A_id: int,
    B_id: int,
) -> np.ndarray:
    device = model.device
    pad_id = tok.pad_token_id or tok.eos_token_id

    max_len = max(len(x) for x in prompt_ids_batch)
    input_ids, attention_mask, lengths = [], [], []

    for ids in prompt_ids_batch:
        lengths.append(len(ids))
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_pos = lengths - 1
    last_logits = out.logits[torch.arange(out.logits.size(0), device=device), last_pos]
    logp = F.log_softmax(last_logits, dim=-1)

    scores = (logp[:, A_id] - logp[:, B_id]).detach().cpu().float().numpy()
    return scores


def build_markdown_report(
    adapter_dir: Optional[Any],
    base_model: str,
    data_file: Path,
    metrics: BinaryMetrics,
    report_text: str,
    threshold: float,
    label_tokens: str,
    pred_pos_rate: float,
) -> str:
    if adapter_dir:
        adapter_str = Path(adapter_dir).as_posix()
    else:
        adapter_str = "None (Base Model)"
    
    md: List[str] = []
    md.append("# Report Valutazione Sarcasmo (Phi-3)\n")
    md.append(f"**Adapter:** `{adapter_str}`  ")
    md.append(f"**Base model:** `{base_model}`  ")
    md.append(f"**Dataset:** `{data_file.as_posix()}`  ")
    md.append(f"**Batch size:** `{BATCH_SIZE}`  ")
    md.append(f"**Max prompt tokens:** `{MAX_PROMPT_TOKENS}`  ")
    md.append(f"**Label scoring:** `{label_tokens}`  ")
    md.append(f"**Threshold (score=logP(A)-logP(B)):** **{threshold:.3f}** ")
    md.append(f"**Predicted positive rate:** **{pred_pos_rate:.4f}**\n")
    md.append("---\n")
    md.append("## Metriche Principali (Classe 'Sarcastic')\n")
    md.append("| Metrica | Valore |")
    md.append("| :--- | :--- |")
    md.append(f"| **Accuracy** | **{metrics.accuracy:.4f}** |")
    md.append(f"| **F1-Score** | **{metrics.f1:.4f}** |")
    md.append(f"| **Precision** | {metrics.precision:.4f} |")
    md.append(f"| **Recall** | {metrics.recall:.4f} |")
    md.append("\n---\n")
    md.append("## Confusion Matrix\n")
    md.append("`Rows=True [0,1], Cols=Pred [0,1]`")
    md.append("")
    md.append(f"- TN={metrics.tn}  FP={metrics.fp}")
    md.append(f"- FN={metrics.fn}  TP={metrics.tp}\n")
    md.append("---\n")
    md.append("## Report Dettagliato\n")
    md.append("```text")
    md.append(report_text)
    md.append("```")
    return "\n".join(md)

def main() -> None:
    rows = load_jsonl(REDDIT_JSONL)
    if not rows:
        raise ValueError(f"No rows found in {REDDIT_JSONL}")

    texts = [r["text"] for r in rows]
    y_true = np.array([int(r["label"]) for r in rows], dtype=int)

    model, tok, base_model = load_phi3_with_lora(ADAPTER_DIR)

    A_ids = tok.encode("A", add_special_tokens=False)
    B_ids = tok.encode("B", add_special_tokens=False)
    if len(A_ids) != 1 or len(B_ids) != 1:
        raise ValueError(f'"A"/"B" must be single token. A={A_ids}, B={B_ids}')
    A_id, B_id = A_ids[0], B_ids[0]

    scores_all = np.zeros((len(texts),), dtype=np.float32)

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch_texts = texts[start:start + BATCH_SIZE]
        prompts = [build_prompt_A(t) for t in batch_texts]

        prompt_ids_batch = []
        for p in prompts:
            ids = tok(
                p,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
            )["input_ids"]
            prompt_ids_batch.append(ids)

        scores = score_batch_next_token_AB(model, tok, prompt_ids_batch, A_id, B_id)
        scores_all[start:start + len(batch_texts)] = scores

    y_pred = (scores_all >= THRESHOLD).astype(int)
    pred_pos_rate = float(y_pred.mean())

    m = binary_metrics(y_true, y_pred)
    rep = classification_report_like(y_true, y_pred)

    md = build_markdown_report(
        adapter_dir=ADAPTER_DIR,
        base_model=base_model,
        data_file=REDDIT_JSONL,
        metrics=m,
        report_text=rep,
        threshold=THRESHOLD,
        label_tokens='score = logP("A") - logP("B")',
        pred_pos_rate=pred_pos_rate,
    )

    REPORT_OUT.write_text(md, encoding="utf-8")
    print(f"Saved report to: {REPORT_OUT.resolve()}")

    with SCORES_OUT.open("w", encoding="utf-8") as f:
        for yt, s in zip(y_true.tolist(), scores_all.tolist()):
            f.write(json.dumps({"label": int(yt), "score": float(s)}, ensure_ascii=False) + "\n")
    print(f"Saved scores to: {SCORES_OUT.resolve()}")


if __name__ == "__main__":
    main()