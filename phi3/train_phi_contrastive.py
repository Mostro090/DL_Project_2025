import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

IGNORE_INDEX = -100


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must define a mapping at the top level.")
    return cfg


def get_single_token_id(tok, s_list=("A", " A")):
    for s in s_list:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s
    raise ValueError(f"Could not find single-token variant for {s_list}. Got: {[(s, tok.encode(s, add_special_tokens=False)) for s in s_list]}")


def get_label_token_ids(tok):
    A_id, A_str = get_single_token_id(tok, ("A", " A"))
    B_id, B_str = get_single_token_id(tok, ("B", " B"))
    return A_id, B_id, A_str, B_str


@dataclass
class DataCollatorCausalLM:
    tokenizer: AutoTokenizer
    ignore_index: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        def pad_list(x, pad_value):
            return x + [pad_value] * (max_len - len(x))

        input_ids = torch.tensor([pad_list(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        attention_mask = torch.tensor([pad_list(f["attention_mask"], 0) for f in features], dtype=torch.long)
        labels = torch.tensor([pad_list(f["labels"], self.ignore_index) for f in features], dtype=torch.long)

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "task" in features[0]:
            out["task"] = [f["task"] for f in features]
        if "gold_label" in features[0]:
            out["gold_label"] = [f["gold_label"] for f in features]
        return out


@dataclass
class DataCollatorPairwiseD:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def pad_2d(seqs, pad_value):
            max_len = max(len(x) for x in seqs)
            out = [x + [pad_value] * (max_len - len(x)) for x in seqs]
            return torch.tensor(out, dtype=torch.long)

        def pad_mask(seqs):
            max_len = max(len(x) for x in seqs)
            out = [x + [0] * (max_len - len(x)) for x in seqs]
            return torch.tensor(out, dtype=torch.long)

        input_ids_s = pad_2d([f["input_ids_s"] for f in features], pad_id)
        attn_s = pad_mask([f["attention_mask_s"] for f in features])

        input_ids_r = pad_2d([f["input_ids_r"] for f in features], pad_id)
        attn_r = pad_mask([f["attention_mask_r"] for f in features])

        return {
            "input_ids_s": input_ids_s,
            "attention_mask_s": attn_s,
            "input_ids_r": input_ids_r,
            "attention_mask_r": attn_r,
        }


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gold_labels: Optional[List[int]],
    pos_weight: float,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, Lm1, V = shift_logits.shape

    loss_tok = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(B, Lm1)

    active = (shift_labels != ignore_index).float()
    w = torch.ones((B,), device=logits.device, dtype=loss_tok.dtype)

    if gold_labels is not None:
        for i, g in enumerate(gold_labels):
            g_val = int(g.item()) if isinstance(g, torch.Tensor) else int(g)
            if g_val == 1:
                w[i] *= float(pos_weight)

    w = w.view(B, 1)
    weighted_sum = (loss_tok * active * w).sum()
    denom = (active * w).sum().clamp_min(1.0)
    return weighted_sum / denom


def score_delta_ab(model, input_ids, attention_mask, A_id, B_id):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    lengths = attention_mask.sum(dim=1)
    last_pos = (lengths - 1).clamp_min(0)
    last_logits = out.logits[torch.arange(out.logits.size(0), device=out.logits.device), last_pos]
    logp = torch.log_softmax(last_logits, dim=-1)
    return logp[:, A_id] - logp[:, B_id]


@torch.no_grad()
def eval_taskA_score(model, tokenizer, ds_val_A, batch_size: int = 8, tau: float = 0.0):
    device = model.device
    A_id, B_id, _, _ = get_label_token_ids(tokenizer)

    y_true, y_pred = [], []
    n = len(ds_val_A)

    for i in range(0, n, batch_size):
        batch = ds_val_A[i:i + batch_size]
        input_ids_col = batch["input_ids"]
        labels_col = batch["labels"]
        gold = batch["gold_label"] if "gold_label" in batch else batch["label"]

        prompts = []
        for ids, labs in zip(input_ids_col, labels_col):
            prompts.append([tid for tid, lab in zip(ids, labs) if lab == IGNORE_INDEX])

        max_len = max(len(p) for p in prompts)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids, attn = [], []
        for p in prompts:
            pad_len = max_len - len(p)
            input_ids.append(p + [pad_id] * pad_len)
            attn.append([1] * len(p) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn = torch.tensor(attn, dtype=torch.long, device=device)

        scores = score_delta_ab(model, input_ids, attn, A_id, B_id).detach().cpu()
        preds = (scores.float() > float(tau)).to(torch.int).tolist()

        y_true.extend(list(gold))
        y_pred.extend(preds)

    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def main(cfg: Dict[str, Any]):
    dataset_dir = cfg["dataset_dir"]
    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]

    seed = int(cfg.get("seed", 42))
    max_steps = int(cfg.get("max_steps", 1000))
    batch_size = int(cfg.get("batch_size", 2))
    grad_accum = int(cfg.get("grad_accum", 8))

    lr = float(cfg.get("lr", 2e-4))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    a_frac = float(cfg.get("a_frac", 0.7))
    eval_every = int(cfg.get("eval_every", 200))
    eval_batch_size = int(cfg.get("eval_batch_size", 8))

    pos_weight = float(cfg.get("pos_weight", 1.0))
    lambda_a = float(cfg.get("lambda_a", 1.0))
    lambda_c = float(cfg.get("lambda_c", 1.0))

    lambda_d = float(cfg.get("lambda_d", cfg.get("lambda_D", 0.5)))
    margin = float(cfg.get("margin", 0.5))
    tau = 0.0

    perf = cfg.get("performance", {})
    grad_ckpt = bool(perf.get("gradient_checkpointing", True))

    lora_cfg = cfg.get("lora", {})
    lora_r = int(lora_cfg.get("r", 16))
    lora_alpha = int(lora_cfg.get("alpha", 32))
    lora_dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    random.seed(seed)
    torch.manual_seed(seed)

    ds = load_from_disk(dataset_dir)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    if "task" in val_ds.column_names:
        val_A = val_ds.filter(lambda x: x["task"] == "A")
    else:
        val_A = val_ds

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    A_id, B_id, A_str, B_str = get_label_token_ids(tok)
    print("Using label tokens:", {"A": A_str, "B": B_str, "A_id": A_id, "B_id": B_id})

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    peft_conf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_conf)
    model.print_trainable_parameters()
    model.train()

    if "task" not in train_ds.column_names:
        raise ValueError("Train dataset must have a 'task' column.")

    train_A = train_ds.filter(lambda x: x["task"] == "A")
    train_D = train_ds.filter(lambda x: x["task"] == "D") if "D" in set(train_ds["task"]) else None

    if len(train_A) == 0:
        raise ValueError("No Task A examples found in train split.")

    collator_A = DataCollatorCausalLM(tok)
    train_loader_A = DataLoader(
        train_A,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator_A,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    train_loader_D = None
    collator_D = None
    if train_D is not None and len(train_D) > 0 and all(k in train_D.column_names for k in ["input_ids_s", "attention_mask_s", "input_ids_r", "attention_mask_r"]):
        collator_D = DataCollatorPairwiseD(tok)
        train_loader_D = DataLoader(
            train_D,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator_D,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(max_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    device = model.device
    optimizer.zero_grad()

    pbar = tqdm(total=max_steps, desc="Fine-tuning (updates)")
    running_loss = 0.0
    update_step = 0
    micro_step = 0

    iterA = iter(train_loader_A)
    iterD = iter(train_loader_D) if train_loader_D is not None else None

    use_pairwise = train_loader_D is not None

    while update_step < max_steps:
        micro_step += 1

        use_A = True
        if use_pairwise:
            use_A = (random.random() < a_frac)

        if use_A:
            try:
                batch = next(iterA)
            except StopIteration:
                iterA = iter(train_loader_A)
                batch = next(iterA)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            gold = batch.get("gold_label", None)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_A = weighted_causal_lm_loss(out.logits, labels, gold, pos_weight, IGNORE_INDEX) * float(lambda_a)
            loss = loss_A / grad_accum

        else:
            try:
                batch = next(iterD)
            except StopIteration:
                iterD = iter(train_loader_D)
                batch = next(iterD)

            ids_s = batch["input_ids_s"].to(device)
            attn_s = batch["attention_mask_s"].to(device)
            ids_r = batch["input_ids_r"].to(device)
            attn_r = batch["attention_mask_r"].to(device)

            delta_s = score_delta_ab(model, ids_s, attn_s, A_id, B_id)
            delta_r = score_delta_ab(model, ids_r, attn_r, A_id, B_id)

            loss_D = torch.relu(float(margin) - (delta_s - delta_r)).mean() * float(lambda_d)
            loss = loss_D / grad_accum

        loss.backward()
        running_loss += loss.item()

        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            update_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{running_loss:.4f}", "a_frac": f"{a_frac:.2f}", "tau": f"{tau:.2f}"})
            running_loss = 0.0

            if update_step % eval_every == 0 or update_step == max_steps:
                model.eval()
                metrics = eval_taskA_score(model, tok, val_A, batch_size=eval_batch_size, tau=tau)
                print(
                    f"\n[VAL @ {update_step}] "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}\n"
                )
                model.train()

                save_path = f"{output_dir}/step_{update_step}"
                model.save_pretrained(save_path)
                tok.save_pretrained(save_path)

    pbar.close()
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    print(f"Done. Saved final adapter to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
