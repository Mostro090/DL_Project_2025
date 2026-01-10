from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import ast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

IGNORE_IDX = -100
MODEL_DIR = Path(f"../runs/modernbert_o2") 
DATASET_DIR = Path(f"../../data/tokenized/o2")
SPLIT = "test"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
REPORT_OUT = f"report_test_modernbert_o2.md"

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
        support_0=support_0, support_1=support_1
    )

def classification_report_like(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    m_pos = binary_metrics(y_true, y_pred)

    tp0, fp0, fn0 = m_pos.tn, m_pos.fn, m_pos.fp
    prec0 = tp0 / max(1, (tp0 + fp0))
    rec0  = tp0 / max(1, (tp0 + fn0))
    f10   = (2 * prec0 * rec0) / max(1e-12, (prec0 + rec0))

    prec1, rec1, f11 = m_pos.precision, m_pos.recall, m_pos.f1

    acc = m_pos.accuracy
    support0, support1 = m_pos.support_0, m_pos.support_1
    total = support0 + support1

    macro_prec = (prec0 + prec1) / 2
    macro_rec  = (rec0 + rec1) / 2
    macro_f1   = (f10 + f11) / 2

    weighted_prec = (prec0 * support0 + prec1 * support1) / max(1, total)
    weighted_rec  = (rec0 * support0 + rec1 * support1) / max(1, total)
    weighted_f1   = (f10 * support0 + f11 * support1) / max(1, total)

    lines = []
    lines.append("                precision    recall  f1-score   support")
    lines.append("")
    lines.append(f"Non Sarcastic   {prec0:9.2f}  {rec0:8.2f}  {f10:8.2f}  {support0:8d}")
    lines.append(f"Sarcastic       {prec1:9.2f}  {rec1:8.2f}  {f11:8.2f}  {support1:8d}")
    lines.append("")
    lines.append(f"accuracy                            {acc:5.2f}  {total:8d}")
    lines.append(f"macro avg       {macro_prec:9.2f}  {macro_rec:8.2f}  {macro_f1:8.2f}  {total:8d}")
    lines.append(f"weighted avg    {weighted_prec:9.2f}  {weighted_rec:8.2f}  {weighted_f1:8.2f}  {total:8d}")
    return "\n".join(lines)

class SarcasmInferenceModel(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        hidden = self.encoder.config.hidden_size
        self.head_score = nn.Linear(hidden, 1)

    def get_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

def load_full_model(model_dir: Path, device: str) -> Tuple[SarcasmInferenceModel, AutoTokenizer]:
    abs_model_dir = model_dir.resolve()
    print(f"Loading encoder/adapter from {abs_model_dir}...")

    if not abs_model_dir.exists():
        raise FileNotFoundError(f"Errore: La cartella {abs_model_dir} non esiste. Hai finito il training?")

    from peft import PeftConfig, PeftModel
    try:
        peft_cfg = PeftConfig.from_pretrained(str(abs_model_dir))
    except Exception as e:
        raise FileNotFoundError(f"Impossibile trovare adapter_config.json in {abs_model_dir}.") from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(abs_model_dir))
    except Exception:
        print(f"⚠️ Tokenizer locale non trovato. Carico il base: {peft_cfg.base_model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)

    base = AutoModel.from_pretrained(peft_cfg.base_model_name_or_path)
    encoder = PeftModel.from_pretrained(base, str(abs_model_dir))
    
    model = SarcasmInferenceModel(encoder=encoder)

    heads_path = abs_model_dir / "head_score.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"Missing head_score.pt in {abs_model_dir}.")

    print(f"Loading classification head from {heads_path}...")
    ckpt = torch.load(str(heads_path), map_location="cpu")
    model.head_score.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def predict_task_a(
    model: SarcasmInferenceModel,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ys: List[np.ndarray] = []
    ls: List[np.ndarray] = []

    for batch in dataloader:
        if "input_ids_a" in batch:
            ids = batch["input_ids_a"].to(device)
            mask = batch["attention_mask_a"].to(device)
            y_batch = batch["label_a"]
        else:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y_batch = batch.get("label_a", batch.get("label", None))

        if y_batch is None: continue

        cls = model.get_cls(ids, mask)
        logits = model.head_score(cls).squeeze(-1) 

        y = y_batch.cpu().numpy()
        log = logits.detach().cpu().numpy()

        mask_valid = y != IGNORE_IDX
        if np.any(mask_valid):
            ys.append(y[mask_valid])
            ls.append(log[mask_valid])

    if not ys:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_true = np.concatenate(ys).astype(int)
    logits = np.concatenate(ls).astype(float)
    return y_true, logits

def threshold_to_preds(logits: np.ndarray, thr: float) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= thr).astype(int)

def build_markdown_report(
    model_name: str,
    file_analyzed: str,
    metrics: BinaryMetrics,
    report_text: str,
) -> str:
    total_valid = metrics.support_0 + metrics.support_1

    md = []
    md.append(f"# Report Valutazione Sarcasmo (Last Checkpoint)\n")
    md.append(f"**Modello:** `{model_name}`  ")
    md.append(f"**File Analizzato:** `{file_analyzed}`  ")
    md.append(f"**Soglia Utilizzata:** **{THRESHOLD:.2f}** (Fixed)\n")
    md.append("---\n")
    md.append("## 1. Metriche Principali (Classe 'Sarcastic')\n")
    md.append("| Metrica | Valore | Descrizione |")
    md.append("| :--- | :--- | :--- |")
    md.append(f"| **Accuracy** | **{metrics.accuracy:.4f}** | Percentuale totale di risposte corrette |")
    md.append(f"| **F1-Score** | **{metrics.f1:.4f}** | Media armonica tra Precision e Recall |")
    md.append(f"| **Precision** | {metrics.precision:.4f} | Quando predice \"Sarcasmo\", quanto spesso ha ragione? |")
    md.append(f"| **Recall** | {metrics.recall:.4f} | Quanto sarcasmo reale riesce a trovare? |")
    md.append("\n---\n")
    md.append("## 3. Report Dettagliato\n")
    md.append("```text")
    md.append(report_text)
    md.append("```")
    return "\n".join(md)

def main():
    print(f"Loading model from: {MODEL_DIR.resolve()}")
    model, tokenizer = load_full_model(MODEL_DIR, device=DEVICE)
    collator = DataCollatorWithPadding(tokenizer)

    print(f"Loading data from: {DATASET_DIR.resolve()}")
    ds = load_from_disk(str(DATASET_DIR))
    
    if SPLIT not in ds:
        raise ValueError(f"Split '{SPLIT}' not found.")

    def make_dl(split_name: str) -> DataLoader:
        d = ds[split_name]
        cols = [c for c in ["input_ids", "attention_mask", "label_a", "input_ids_a", "attention_mask_a"] if c in d.column_names]
        d.set_format(type="torch", columns=cols)
        return DataLoader(d, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    print(f"\n--- VALUTAZIONE SU {SPLIT} con Soglia Fissa {THRESHOLD:.2f} ---")
    dl_eval = make_dl(SPLIT)
    y_true, logits = predict_task_a(model, dl_eval, device=DEVICE)
    
    if len(y_true) == 0:
        print(f"Nessun esempio valido trovato.")
        return

    y_pred = threshold_to_preds(logits, THRESHOLD)
    m = binary_metrics(y_true, y_pred)
    rep = classification_report_like(y_true, y_pred)

    model_name = MODEL_DIR.name
    file_analyzed = f"{DATASET_DIR.name}:{SPLIT}"
    md = build_markdown_report(model_name=model_name, file_analyzed=file_analyzed, metrics=m, report_text=rep)

    print("\n" + md + "\n")

    if REPORT_OUT:
        Path(REPORT_OUT).write_text(md, encoding="utf-8")
        print(f"Report saved to: {REPORT_OUT}")

if __name__ == "__main__":
    main()