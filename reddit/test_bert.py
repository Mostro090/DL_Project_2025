from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftConfig, PeftModel

MODEL_DIR = Path("../results/runs/modernbert_o2_second") 
REDDIT_JSONL = Path("./reddit_holdout_20k.jsonl")     

BATCH_SIZE = 32
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.75
REPORT_OUT = Path("report_modernbert.md")


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_markdown_report(
    model_name: str,
    base_model: str,
    file_analyzed: str,
    metrics: BinaryMetrics,
    report_text: str,
    threshold: float,
    pred_pos_rate: float,
) -> str:
    md: List[str] = []
    md.append("# Report Valutazione Sarcasmo (ModernBERT LoRA)\n")
    md.append(f"**Model dir:** `{model_name}`  ")
    md.append(f"**Base model:** `{base_model}`  ")
    md.append(f"**Dataset:** `{file_analyzed}`  ")
    md.append(f"**Batch size:** `{BATCH_SIZE}`  ")
    md.append(f"**Max length:** `{MAX_LEN}`  ")
    md.append(f"**Threshold (sigmoid(logit) >= t):** **{threshold:.2f}**  ")
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


class SarcasmInferenceModel(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size
        self.head_score = nn.Linear(hidden, 1)

    def get_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


def load_full_model(model_dir: Path, device: str) -> Tuple[SarcasmInferenceModel, AutoTokenizer, str]:
    abs_dir = model_dir.resolve()
    if not abs_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {abs_dir}")

    peft_cfg = PeftConfig.from_pretrained(str(abs_dir))
    base_name = peft_cfg.base_model_name_or_path

    try:
        tok = AutoTokenizer.from_pretrained(str(abs_dir), use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)

    base = AutoModel.from_pretrained(base_name)
    encoder = PeftModel.from_pretrained(base, str(abs_dir))

    model = SarcasmInferenceModel(encoder=encoder)

    head_path = abs_dir / "head_score.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Missing head_score.pt in {abs_dir}")

    ckpt = torch.load(str(head_path), map_location="cpu")
    model.head_score.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model, tok, base_name


def load_jsonl(path: Path) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            texts.append(ex["text"])
            labels.append(int(ex["label"]))
    return texts, np.array(labels, dtype=int)


@torch.no_grad()
def predict_logits(model: SarcasmInferenceModel, tok: AutoTokenizer, texts: List[str], device: str) -> np.ndarray:
    logits_all: List[np.ndarray] = []

    running_pos = 0
    seen = 0

    pbar = tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring", unit="batch", dynamic_ncols=True)
    for start in pbar:
        batch_texts = texts[start:start + BATCH_SIZE]
        inputs = tok(
            [t for t in batch_texts],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        cls = model.get_cls(ids, mask)
        logits = torch.clamp(model.head_score(cls), -10, 10).squeeze(-1)

        log_np = logits.detach().cpu().numpy().astype(float)
        logits_all.append(log_np)

        probs = 1.0 / (1.0 + np.exp(-log_np))
        batch_pred = (probs >= THRESHOLD).astype(int)
        running_pos += int(batch_pred.sum())
        seen += len(batch_pred)

        p50 = float(np.quantile(probs, 0.50))
        p90 = float(np.quantile(probs, 0.90))
        pos_rate = running_pos / max(1, seen)

        pbar.set_postfix(pos_rate=f"{pos_rate:.3f}", p50=f"{p50:.3f}", p90=f"{p90:.3f}")

    return np.concatenate(logits_all, axis=0)


def main() -> None:
    model, tok, base_name = load_full_model(MODEL_DIR, device=DEVICE)
    texts, y_true = load_jsonl(REDDIT_JSONL)

    logits = predict_logits(model, tok, texts, device=DEVICE)
    probs = sigmoid(logits)
    y_pred = (probs >= THRESHOLD).astype(int)

    pred_pos_rate = float(y_pred.mean())
    m = binary_metrics(y_true, y_pred)
    rep = classification_report_like(y_true, y_pred)

    md = build_markdown_report(
        model_name=MODEL_DIR.as_posix(),
        base_model=base_name,
        file_analyzed=REDDIT_JSONL.as_posix(),
        metrics=m,
        report_text=rep,
        threshold=THRESHOLD,
        pred_pos_rate=pred_pos_rate,
    )

    REPORT_OUT.write_text(md, encoding="utf-8")
    print(f"Saved report to: {REPORT_OUT.resolve()}")


if __name__ == "__main__":
    main()
