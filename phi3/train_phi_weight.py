import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

IGNORE_INDEX = -100


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must define a mapping at the top level.")
    return cfg


@dataclass
class DataCollatorCausalLM:
    tokenizer: AutoTokenizer
    ignore_index: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

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


class MixedTaskBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, a_frac: float, seed: int, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.a_frac = a_frac
        self.seed = seed
        self.drop_last = drop_last

        if "task" not in dataset.column_names:
            raise ValueError("Train dataset must have a 'task' column with values 'A'/'C'.")

        tasks = dataset["task"]
        self.idx_A = [i for i, t in enumerate(tasks) if t == "A"]
        self.idx_C = [i for i, t in enumerate(tasks) if t == "C"]

        if not self.idx_A or not self.idx_C:
            raise ValueError(f"Need both tasks in train split. Found A={len(self.idx_A)}, C={len(self.idx_C)}")

        self.nA = max(1, int(round(self.batch_size * self.a_frac)))
        self.nC = self.batch_size - self.nA
        if self.nC == 0:
            self.nC = 1
            self.nA = self.batch_size - 1

    def __iter__(self):
        rng = random.Random(self.seed)
        A = self.idx_A[:]
        C = self.idx_C[:]
        rng.shuffle(A)
        rng.shuffle(C)

        pA, pC = 0, 0
        n_batches = len(A) // self.nA if self.drop_last else math.ceil(len(A) / self.nA)

        for _ in range(n_batches):
            batch = []
            for _ in range(self.nA):
                if pA >= len(A):
                    rng.shuffle(A)
                    pA = 0
                batch.append(A[pA]); pA += 1

            for _ in range(self.nC):
                if pC >= len(C):
                    rng.shuffle(C)
                    pC = 0
                batch.append(C[pC]); pC += 1

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.idx_A) // self.nA
        return math.ceil(len(self.idx_A) / self.nA)


@torch.no_grad()
def eval_taskA_score(model, tokenizer, ds_val_A, batch_size: int = 8):
    device = model.device

    A_ids = tokenizer.encode("A", add_special_tokens=False)
    B_ids = tokenizer.encode("B", add_special_tokens=False)

    if len(A_ids) != 1 or len(B_ids) != 1:
        raise ValueError(f'Labels not single token: "A"={A_ids}, "B"={B_ids}')

    A_id, B_id = A_ids[0], B_ids[0]

    y_true, y_pred = [], []
    n = len(ds_val_A)

    for i in range(0, n, batch_size):
        batch = ds_val_A[i:i + batch_size]
        input_ids_col = batch["input_ids"]
        labels_col = batch["labels"]
        gold = batch["gold_label"] if "gold_label" in batch else batch["label"]

        prompts = []
        for ids, labs in zip(input_ids_col, labels_col):
            p = [tid for tid, lab in zip(ids, labs) if lab == IGNORE_INDEX]
            prompts.append(p)

        max_len = max(len(p) for p in prompts)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids, attn, lengths = [], [], []
        for p in prompts:
            lengths.append(len(p))
            pad_len = max_len - len(p)
            input_ids.append(p + [pad_id] * pad_len)
            attn.append([1] * len(p) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn = torch.tensor(attn, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attn)
        last_pos = lengths - 1
        last_logits = out.logits[torch.arange(out.logits.size(0), device=device), last_pos]
        logp = torch.log_softmax(last_logits, dim=-1)

        scores = (logp[:, A_id] - logp[:, B_id]).detach().cpu()

        preds = (scores.float() > 0.0).to(torch.int).tolist()

        y_true.extend(list(gold))
        y_pred.extend(preds)

    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def weighted_causal_lm_loss(
    logits: torch.Tensor,          # [B, L, V]
    labels: torch.Tensor,          # [B, L]
    tasks: List[str] | None,       # len B
    gold_labels: List[int] | None, # len B
    pos_weight: float,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Loss token-level CausalLM con peso maggiore SOLO per esempi task A positivi (gold_label==1).
    Task C e task A negativi restano peso 1.
    Normalizzazione per somma pesi sui token attivi -> scala stabile.
    """
    B, L, V = logits.shape

    loss_tok = F.cross_entropy(
        logits.view(B * L, V),
        labels.view(B * L),
        reduction="none",
        ignore_index=ignore_index,
    ).view(B, L)

    active = (labels != ignore_index).float()  # [B, L]

    w = torch.ones((B,), device=logits.device, dtype=loss_tok.dtype)

    if tasks is not None and gold_labels is not None:
        for i, (t, g) in enumerate(zip(tasks, gold_labels)):
            if t == "A" and int(g) == 1:
                w[i] = float(pos_weight)

    w = w.view(B, 1)
    weighted_sum = (loss_tok * active * w).sum()
    denom = (active * w).sum().clamp_min(1.0)
    return weighted_sum / denom


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

    # nuovo: peso positivi task A
    pos_weight = float(cfg.get("pos_weight", 3.0))

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

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

    collator = DataCollatorCausalLM(tok)

    sampler = MixedTaskBatchSampler(
        train_ds,
        batch_size=batch_size,
        a_frac=a_frac,
        seed=seed,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(max_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    device = model.device
    optimizer.zero_grad()

    pbar = tqdm(total=max_steps, desc="Fine-tuning (updates)")
    running_loss = 0.0
    update_step = 0
    micro_step = 0

    while update_step < max_steps:
        for batch in train_loader:
            micro_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            tasks = batch.get("task", None)          # lista stringhe (A/C)
            gold = batch.get("gold_label", None)     # lista int (0/1) per A

            # forward senza labels, loss calcolato manualmente per pesare i positivi del task A
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = weighted_causal_lm_loss(
                logits=out.logits,
                labels=labels,
                tasks=tasks,
                gold_labels=gold,
                pos_weight=pos_weight,
                ignore_index=IGNORE_INDEX,
            ) / grad_accum

            loss.backward()
            running_loss += loss.item()

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                update_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{running_loss:.4f}", "pos_w": f"{pos_weight:.2f}"})
                running_loss = 0.0

                if update_step % eval_every == 0 or update_step == max_steps:
                    model.eval()
                    metrics = eval_taskA_score(
                        model, tok, val_A, batch_size=eval_batch_size
                    )
                    print(
                        f"\n[VAL @ {update_step}] "
                        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                        f"(pos_weight={pos_weight:.2f})\n"
                    )
                    model.train()

                    save_path = f"{output_dir}/step_{update_step}"
                    model.save_pretrained(save_path)
                    tok.save_pretrained(save_path)

                if update_step >= max_steps:
                    break

        if update_step >= max_steps:
            break

    pbar.close()
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tok.save_pretrained(final_path)
    print(f"Done. Saved final adapter to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
