from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel, PeftConfig
from tqdm import tqdm

ADAPTER_DIR = Path("../runs/phi3_o2/step_400")
DATASET_DIR = Path("../../data/phi3_dataset")
SPLIT = "test"
BATCH_SIZE = 16
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
THRESHOLD = 1.10
REPORT_OUT = "report_phi3_o2.md"
IGNORE_IDX = -100

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
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
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
    lines.append(
        f"macro avg       {macro_prec:9.2f}  {macro_rec:8.2f}  {macro_f1:8.2f}  {total:8d}"
    )
    lines.append(
        f"weighted avg    {weighted_prec:9.2f}  {weighted_rec:8.2f}  {weighted_f1:8.2f}  {total:8d}"
    )
    return "\n".join(lines)

def load_phi3_lora(adapter_path: Path, device: str):
    abs_adapter_path = adapter_path.resolve()
    if not abs_adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {abs_adapter_path}")
    
    peft_cfg = PeftConfig.from_pretrained(str(abs_adapter_path))
    base_model_name = peft_cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, str(abs_adapter_path))
    model.eval()
    
    return model, tokenizer

def extract_prompt_ids(input_ids_col, labels_col):
    prompts = []
    for ids, labs in zip(input_ids_col, labels_col):
        prompt_ids = [tid for tid, lab in zip(ids, labs) if lab == IGNORE_IDX]
        prompts.append(prompt_ids)
    return prompts

def list_collator(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

@torch.no_grad()
def predict_phi3_logits(model, tokenizer, dataloader, device):
    ys_true = []
    scores_diff = []

    token_A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    token_B_id = tokenizer.encode("B", add_special_tokens=False)[0]
    pad_id = tokenizer.pad_token_id

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        prompt_ids_list = extract_prompt_ids(input_ids, labels)
        
        if "gold_label" in batch:
            golds = batch["gold_label"]
        elif "label" in batch:
            golds = batch["label"]
        else:
            continue

        max_len = max(len(p) for p in prompt_ids_list)
        
        padded_input_ids = []
        attention_masks = []
        lengths = []

        for p in prompt_ids_list:
            l = len(p)
            lengths.append(l)
            pad_len = max_len - l
            padded_input_ids.append(p + [pad_id] * pad_len)
            attention_masks.append([1] * l + [0] * pad_len)

        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=device)

        outputs = model(input_ids=input_tensor, attention_mask=mask_tensor, use_cache=False)
        
        last_token_indices = torch.tensor(lengths, device=device) - 1
        relevant_logits = outputs.logits[torch.arange(outputs.logits.shape[0]), last_token_indices]
        
        log_probs = F.log_softmax(relevant_logits, dim=-1)

        score_A = log_probs[:, token_A_id]
        score_B = log_probs[:, token_B_id]
        diff = score_A - score_B

        ys_true.extend(golds)
        scores_diff.extend(diff.cpu().numpy())

    return np.array(ys_true), np.array(scores_diff)

def build_markdown_report(
    model_name: str,
    file_analyzed: str,
    metrics: BinaryMetrics,
    report_text: str,
    confusion_matrix_str: str
) -> str:
    md: List[str] = []
    md.append("# Report Valutazione Sarcasmo (Phi-3 LoRA)\n")
    md.append(f"**Modello Adapter:** `{model_name}`  ")
    md.append(f"**File Analizzato:** `{file_analyzed}`  ")
    md.append(f"**Soglia (Logits A-B):** **{THRESHOLD:.2f}**\n")
    md.append("---\n")
    md.append("## Metriche Principali (Classe 'Sarcastic')\n")
    md.append("| Metrica | Valore |")
    md.append("| :--- | :--- |")
    md.append(f"| **Accuracy** | **{metrics.accuracy:.4f}** |")
    md.append(f"| **F1-Score** | **{metrics.f1:.4f}** |")
    md.append(f"| **Precision** | {metrics.precision:.4f} |")
    md.append(f"| **Recall** | {metrics.recall:.4f} |")
    md.append("\n---\n")
    md.append("## Matrice di Confusione\n")
    md.append("```text")
    md.append(confusion_matrix_str)
    md.append("```\n")
    md.append("## Report Dettagliato\n")
    md.append("```text")
    md.append(report_text)
    md.append("```")
    return "\n".join(md)

def main() -> None:
    adapter_path = ADAPTER_DIR
    model, tokenizer = load_phi3_lora(adapter_path, device=DEVICE)

    ds = load_from_disk(str(DATASET_DIR))

    if SPLIT not in ds:
        raise ValueError(f"Split '{SPLIT}' not found in dataset.")
    
    val_ds = ds[SPLIT]
    if "task" in val_ds.column_names:
        val_ds = val_ds.filter(lambda x: x["task"] == "A")

    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=list_collator)

    y_true, scores_diff = predict_phi3_logits(model, tokenizer, val_dl, device=DEVICE)

    if len(y_true) == 0:
        return

    y_pred = (scores_diff > THRESHOLD).astype(int)

    m = binary_metrics(y_true, y_pred)
    rep = classification_report_like(y_true, y_pred)
    
    cm_str = f"[[{m.tn}  {m.fp}]\n [{m.fn}  {m.tp}]]"

    model_name = adapter_path.name
    file_analyzed = f"{DATASET_DIR.name}:{SPLIT}"
    md = build_markdown_report(
        model_name=model_name,
        file_analyzed=file_analyzed,
        metrics=m,
        report_text=rep,
        confusion_matrix_str=cm_str
    )

    print("\n" + md + "\n")

    if REPORT_OUT:
        Path(REPORT_OUT).write_text(md, encoding="utf-8")

if __name__ == "__main__":
    main()