from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Iterator, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

IGNORE_IDX = -100

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


class SarcasmModel(nn.Module):
    def __init__(self, model_id: str, peft_cfg: dict):
        super().__init__()
        base = AutoModel.from_pretrained(model_id)

        if peft_cfg.get("enable", True):
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(peft_cfg.get("lora_r", 8)),
                lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
                lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
                target_modules=peft_cfg.get("target_modules", ["query", "value"]),
            )
            self.encoder = get_peft_model(base, lora_config)
        else:
            self.encoder = base

        hidden = self.encoder.config.hidden_size
        self.head_a = nn.Linear(hidden, 1)  
        self.head_b = nn.Linear(hidden, 2)  

    def get_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


class MixedABBatchSampler:
    def __init__(self, idx_a: List[int], idx_b: List[int], batch_size: int, seed: int = 42):
        assert batch_size % 2 == 0, "batch_size deve essere pari per fare metà A e metà B"
        self.idx_a = list(idx_a)
        self.idx_b = list(idx_b)
        self.bs = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return int(np.ceil(len(self.idx_a) / self.half))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        a = np.array(self.idx_a)
        b = np.array(self.idx_b)

        rng.shuffle(a)
        rng.shuffle(b)

        b_ptr = 0
        b_len = len(b)

        for start in range(0, len(a), self.half):
            a_batch = a[start:start + self.half].tolist()
            if len(a_batch) < self.half:
                break

            if b_len == 0:
                raise ValueError("Non ci sono esempi di task B nel training set.")

            if b_ptr + self.half <= b_len:
                b_batch = b[b_ptr:b_ptr + self.half].tolist()
                b_ptr += self.half
            else:
                tail = b[b_ptr:].tolist()
                need = self.half - len(tail)
                b_batch = tail + b[:need].tolist()
                b_ptr = need

            batch = a_batch + b_batch
            rng.shuffle(batch) 
            yield batch


@torch.no_grad()
def evaluate(model: SarcasmModel, dlA_val: DataLoader, dlB_val: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    ys, ps = [], []
    for batch in dlA_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        cls = model.get_cls(batch["input_ids"], batch["attention_mask"])
        logits = model.head_a(cls).squeeze(-1)
        pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        ys.append(batch["label_a"].cpu().numpy())
        ps.append(pred)
    f1_a = compute_f1(np.concatenate(ys), np.concatenate(ps)) if ys else 0.0

    ys, ps = [], []
    for batch in dlB_val:
        batch = {k: v.to(device) for k, v in batch.items()}
        cls = model.get_cls(batch["input_ids"], batch["attention_mask"])
        logits = model.head_b(cls)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        ys.append(batch["label_b"].cpu().numpy())
        ps.append(pred)
    acc_b = float(np.mean(np.concatenate(ys) == np.concatenate(ps))) if ys else 0.0

    return {"A_f1": float(f1_a), "B_acc": acc_b, "score": float(f1_a + 0.2 * acc_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    device_str = cfg["training"].get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    tokenized_dir = cfg["data"]["tokenized_dir"]
    model_id = cfg["model"]["model_id"]

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.0))
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.0))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    lambda_a = float(cfg["training"].get("lambda_a", 1.0))
    lambda_b = float(cfg["training"].get("lambda_b", 0.3))
    num_workers = int(cfg["training"].get("num_workers", 2))

    ds = load_from_disk(tokenized_dir)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    train_tasks = train_ds["task"]
    idx_a = [i for i, t in enumerate(train_tasks) if t == "A"]
    idx_b = [i for i, t in enumerate(train_tasks) if t == "B"]

    val_tasks = val_ds["task"]
    val_idx_a = [i for i, t in enumerate(val_tasks) if t == "A"]
    val_idx_b = [i for i, t in enumerate(val_tasks) if t == "B"]

    cols = ["input_ids", "attention_mask", "label_a", "label_b"]
    train_ds.set_format(type="torch", columns=[c for c in cols if c in train_ds.column_names])
    val_ds.set_format(type="torch", columns=[c for c in cols if c in val_ds.column_names])

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    collator = DataCollatorWithPadding(tokenizer)

    mix_strategy = cfg["training"].get("mix_strategy", "deterministic_concat")
    if mix_strategy != "deterministic_concat":
        raise ValueError("Questo script implementa mix_strategy=deterministic_concat.")

    batch_sampler = MixedABBatchSampler(idx_a, idx_b, batch_size=batch_size, seed=int(cfg.get("seed", 42)))
    dl_train = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    valA = val_ds.select(val_idx_a)
    valB = val_ds.select(val_idx_b)
    valA.set_format(type="torch", columns=[c for c in cols if c in valA.column_names])
    valB.set_format(type="torch", columns=[c for c in cols if c in valB.column_names])

    dlA_val = DataLoader(valA, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator)
    dlB_val = DataLoader(valB, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator)

    pos_weight_cfg = float(cfg["training"].get("pos_weight_a", -1))
    if pos_weight_cfg > 0:
        pos_weight = torch.tensor([pos_weight_cfg], device=device)
    else:
        y_a = np.array(train_ds["label_a"])
        mask = y_a != IGNORE_IDX
        y = y_a[mask]
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)

    model = SarcasmModel(model_id=model_id, peft_cfg=cfg.get("peft", {})).to(device)
    if hasattr(model.encoder, "print_trainable_parameters"):
        model.encoder.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    total_steps = len(dl_train) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            cls = model.get_cls(batch["input_ids"], batch["attention_mask"])

            mask_a = (batch["label_a"] != IGNORE_IDX)
            loss_a = torch.tensor(0.0, device=device)
            if mask_a.any():
                logits_a = model.head_a(cls[mask_a]).squeeze(-1)
                y_a = batch["label_a"][mask_a].float()
                loss_a = F.binary_cross_entropy_with_logits(logits_a, y_a, pos_weight=pos_weight)

            mask_b = (batch["label_b"] != IGNORE_IDX)
            loss_b = torch.tensor(0.0, device=device)
            if mask_b.any():
                logits_b = model.head_b(cls[mask_b])
                y_b = batch["label_b"][mask_b].long()
                loss_b = F.cross_entropy(logits_b, y_b)

            loss = lambda_a * loss_a + lambda_b * loss_b

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, dlA_val, dlB_val, device)
        avg_loss = total_loss / max(1, len(dl_train))
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  A_f1: {metrics['A_f1']:.4f} | B_acc: {metrics['B_acc']:.4f} | Score: {metrics['score']:.4f}")

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            if hasattr(model.encoder, "save_pretrained"):
                model.encoder.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            torch.save({"head_a": model.head_a.state_dict(), "head_b": model.head_b.state_dict()}, out_dir / "heads.pt")
            (out_dir / "metrics_best.txt").write_text(str(metrics), encoding="utf-8")
            print(f"  Saved best to {out_dir}")

    print(f"\nDone! Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
