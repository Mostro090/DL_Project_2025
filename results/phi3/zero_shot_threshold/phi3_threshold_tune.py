import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

IGNORE_INDEX = -100  # non usato qui, ma lasciato per coerenza


# -------------------------
# I/O
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_task_A(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("task") == "A" and "text" in r and "label" in r:
            out.append({"text": r["text"], "label": int(r["label"])})
    return out


def save_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# -------------------------
# Prompt (stessa logica)
# -------------------------
def build_prompt_A(text: str) -> str:
    return (
        "Classify the following text.\n"
        'Reply ONLY with "A" if it is sarcastic, or "B" if it is NOT sarcastic.\n'
        f"Text: {text}\n"
        "Answer: "
    )


def make_prompt_text(
    tok: AutoTokenizer,
    prompt: str,
    use_chat_template: bool,
    system_msg: str = "You are a helpful assistant.",
) -> str:
    if use_chat_template:
        messages_prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        return tok.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt


# -------------------------
# Report + confusion matrix (stile sklearn)
# -------------------------
def confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # [[tn, fp],
    #  [fn, tp]]
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _prf_support_for_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> Tuple[float, float, float, int]:
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())
    support = int((y_true == cls).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1), support


def build_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    p0, r0, f10, s0 = _prf_support_for_class(y_true, y_pred, 0)
    p1, r1, f11, s1 = _prf_support_for_class(y_true, y_pred, 1)

    acc = float((y_true == y_pred).mean())
    total = int(len(y_true))

    macro_p = (p0 + p1) / 2.0
    macro_r = (r0 + r1) / 2.0
    macro_f1 = (f10 + f11) / 2.0

    # weighted
    weighted_p = (p0 * s0 + p1 * s1) / total if total > 0 else 0.0
    weighted_r = (r0 * s0 + r1 * s1) / total if total > 0 else 0.0
    weighted_f1 = (f10 * s0 + f11 * s1) / total if total > 0 else 0.0

    # formattazione simile a sklearn
    lines = []
    lines.append("              precision    recall  f1-score   support")
    lines.append("")
    lines.append(f"           0     {p0:0.4f}    {r0:0.4f}    {f10:0.4f}      {s0:d}")
    lines.append(f"           1     {p1:0.4f}    {r1:0.4f}    {f11:0.4f}      {s1:d}")
    lines.append("")
    lines.append(f"    accuracy                         {acc:0.4f}      {total:d}")
    lines.append(f"   macro avg     {macro_p:0.4f}    {macro_r:0.4f}    {macro_f1:0.4f}      {total:d}")
    lines.append(f"weighted avg     {weighted_p:0.4f}    {weighted_r:0.4f}    {weighted_f1:0.4f}      {total:d}")
    return "\n".join(lines)


def compute_metrics_for_tuning(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # comodo per scegliere la soglia
    p1, r1, f11, _ = _prf_support_for_class(y_true, y_pred, 1)
    p0, r0, f10, _ = _prf_support_for_class(y_true, y_pred, 0)
    acc = float((y_true == y_pred).mean())
    macro_f1 = (f10 + f11) / 2.0
    return {"accuracy": acc, "f1_pos": f11, "macro_f1": macro_f1}


# -------------------------
# Scoring zero-shot: p(A) vs p(B)
# (robusto multi-token)
# -------------------------
@torch.no_grad()
def score_two_candidates_logprobs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_ids: torch.Tensor,         # [1, L]
    candA_ids: List[int],
    candB_ids: List[int],
) -> Tuple[float, float]:
    device = prompt_ids.device
    out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values

    def seq_logprob(cand_ids: List[int]) -> float:
        lp = 0.0
        local_past = past

        logits_last = out.logits[:, -1, :]
        logp_next = torch.log_softmax(logits_last, dim=-1)

        first = cand_ids[0]
        lp += float(logp_next[0, first].item())

        for tid in cand_ids[1:]:
            step = model(
                input_ids=torch.tensor([[first]], device=device),
                use_cache=True,
                past_key_values=local_past
            )
            local_past = step.past_key_values
            logits = step.logits[:, -1, :]
            logp_next = torch.log_softmax(logits, dim=-1)
            lp += float(logp_next[0, tid].item())
            first = tid

        return lp

    logpA = seq_logprob(candA_ids)
    logpB = seq_logprob(candB_ids)
    return logpA, logpB


@torch.no_grad()
def compute_pA_for_examples(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    examples: List[Dict[str, Any]],
    max_len: int,
    use_chat_template: bool,
    system_msg: str,
    batch_size: int = 16,
) -> np.ndarray:
    device = next(model.parameters()).device

    candA = tok.encode("A", add_special_tokens=False)
    candB = tok.encode("B", add_special_tokens=False)

    # Fast path: 1 token ciascuno => batch
    if len(candA) == 1 and len(candB) == 1:
        a_id, b_id = candA[0], candB[0]
        pAs: List[float] = []

        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            prompts = [
                make_prompt_text(tok, build_prompt_A(ex["text"]), use_chat_template, system_msg)
                for ex in chunk
            ]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            ).to(device)

            out = model(**enc)
            attn = enc["attention_mask"]
            last_pos = attn.sum(dim=1) - 1
            logits_last = out.logits[torch.arange(out.logits.size(0), device=device), last_pos, :]

            logits_ab = torch.stack([logits_last[:, a_id], logits_last[:, b_id]], dim=-1)
            probs_ab = torch.softmax(logits_ab, dim=-1)
            pA_batch = probs_ab[:, 0].detach().cpu().numpy().tolist()
            pAs.extend(pA_batch)

        return np.array(pAs, dtype=np.float64)

    # Fallback robusto: multi-token
    pAs = []
    for ex in examples:
        prompt = make_prompt_text(tok, build_prompt_A(ex["text"]), use_chat_template, system_msg)
        prompt_ids = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )["input_ids"].to(device)

        logpA, logpB = score_two_candidates_logprobs(model, tok, prompt_ids, candA, candB)

        m = max(logpA, logpB)
        pA = np.exp(logpA - m) / (np.exp(logpA - m) + np.exp(logpB - m))
        pAs.append(float(pA))

    return np.array(pAs, dtype=np.float64)


# -------------------------
# Threshold tuning
# -------------------------
def tune_threshold(
    pA: np.ndarray,
    y: np.ndarray,
    metric: str = "f1_pos",
) -> Tuple[float, Dict[str, float]]:
    uniq = np.unique(pA)
    thresholds = np.concatenate(([0.0], uniq, [1.0]))

    best_t = 0.5
    best_metrics = None
    best_score = -1.0

    for t in thresholds:
        y_pred = (pA >= t).astype(np.int64)
        mets = compute_metrics_for_tuning(y, y_pred)
        score = mets.get(metric, mets["f1_pos"])

        if (score > best_score) or (score == best_score and t < best_t):
            best_score = score
            best_t = float(t)
            best_metrics = mets

    assert best_metrics is not None
    return best_t, best_metrics


# -------------------------
# Main
# -------------------------
def main() -> None:
    # ---- variabili locali ----
    jsonl_dir = "phi3_jsonl"
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    max_len = 1024
    use_chat_template = True
    system_msg = "You are a helpful assistant."
    batch_size = 16

    tune_metric = "f1_pos"  # "accuracy" | "f1_pos" | "macro_f1"
    report_path = "threshold_report.md"

    # ---- load data ----
    p = Path(jsonl_dir)
    train_rows = filter_task_A(load_jsonl(str(p / "train.jsonl")))
    val_rows = filter_task_A(load_jsonl(str(p / "validation.jsonl")))
    test_rows = filter_task_A(load_jsonl(str(p / "test.jsonl")))

    trainval = train_rows + val_rows
    y_trainval = np.array([ex["label"] for ex in trainval], dtype=np.int64)
    y_test = np.array([ex["label"] for ex in test_rows], dtype=np.int64)

    # ---- model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if (device == "cuda") else None),
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- compute pA ----
    pA_trainval = compute_pA_for_examples(
        model=model,
        tok=tok,
        examples=trainval,
        max_len=max_len,
        use_chat_template=use_chat_template,
        system_msg=system_msg,
        batch_size=batch_size,
    )

    # ---- tune threshold ----
    best_t, mets_trainval = tune_threshold(pA_trainval, y_trainval, metric=tune_metric)
    y_pred_trainval = (pA_trainval >= best_t).astype(np.int64)

    # ---- test ----
    pA_test = compute_pA_for_examples(
        model=model,
        tok=tok,
        examples=test_rows,
        max_len=max_len,
        use_chat_template=use_chat_template,
        system_msg=system_msg,
        batch_size=batch_size,
    )
    y_pred_test = (pA_test >= best_t).astype(np.int64)

    # ---- build reports ----
    cm_trainval = confusion_matrix_2x2(y_trainval, y_pred_trainval)
    cm_test = confusion_matrix_2x2(y_test, y_pred_test)

    rep_trainval = build_classification_report_text(y_trainval, y_pred_trainval)
    rep_test = build_classification_report_text(y_test, y_pred_test)

    md = []
    md.append("# Baseline Results (Zero-shot, Task A)")
    md.append("")
    md.append(f"- Model: `{model_name}`")
    md.append(f"- jsonl_dir: `{jsonl_dir}`")
    md.append(f"- Prompting: `{'chat_template' if use_chat_template else 'plain'}`")
    md.append(f"- max_len: `{max_len}`")
    md.append(f"- Tuned threshold on: `train+validation` (task A only)")
    md.append(f"- Metric optimized: `{tune_metric}`")
    md.append(f"- Best threshold: `{best_t:.6f}`")
    md.append("")
    md.append("## Train+Validation (at best threshold)")
    md.append("")
    md.append("```")
    md.append(rep_trainval)
    md.append("```")
    md.append("")
    md.append("```")
    md.append(str(cm_trainval))
    md.append("```")
    md.append("")
    md.append("## Test (zero-shot, at tuned threshold)")
    md.append("")
    md.append("```")
    md.append(rep_test)
    md.append("```")
    md.append("")
    md.append("```")
    md.append(str(cm_test))
    md.append("```")
    md.append("")

    save_text(report_path, "\n".join(md))
    print(f"Saved report to: {Path(report_path).resolve()}")
    print(f"Best threshold: {best_t:.6f}")
    print("Train+Val metrics (tuning view):", mets_trainval)


if __name__ == "__main__":
    main()
