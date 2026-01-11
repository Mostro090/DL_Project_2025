from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Iterator, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_curve

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_torch_dtype(s: str | None) -> Optional[torch.dtype]:
    if s is None:
        return None
    s = str(s).lower()
    if s in ("none", "null"):
        return None
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("auto",):
        return None
    raise ValueError(f"Unsupported torch_dtype: {s}")

def pick_device(requested: str) -> torch.device:
    req = str(requested).lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")

class SmartMixCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_a = [f for f in features if f["task"] == "A"]
        batch_b = [f for f in features if f["task"] == "B"]

        out: Dict[str, Any] = {}

        if batch_a:
            input_ids = [f["input_ids"] for f in batch_a]
            padded = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
            out["input_ids_a"] = padded["input_ids"]
            out["attention_mask_a"] = padded["attention_mask"]
            out["label_a"] = torch.tensor([f["label_a"] for f in batch_a], dtype=torch.float)

        if batch_b:
            ids_sarc = [f["input_ids_sarc"] for f in batch_b]
            pad_sarc = self.tokenizer.pad({"input_ids": ids_sarc}, padding=True, return_tensors="pt")
            out["input_ids_b_sarc"] = pad_sarc["input_ids"]
            out["mask_b_sarc"] = pad_sarc["attention_mask"]

            ids_reph = [f["input_ids_reph"] for f in batch_b]
            pad_reph = self.tokenizer.pad({"input_ids": ids_reph}, padding=True, return_tensors="pt")
            out["input_ids_b_reph"] = pad_reph["input_ids"]
            out["mask_b_reph"] = pad_reph["attention_mask"]

        return out

class SarcasmModel(nn.Module):
    def __init__(self, model_id: str, trust_remote_code: bool = True, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, dtype=dtype)
        
        for p in self.encoder.parameters():
            p.requires_grad = True
            
        hidden = self.encoder.config.hidden_size
        self.head_score = nn.Linear(hidden, 1)
        
        enc_dtype = next(self.encoder.parameters()).dtype
        self.head_score = self.head_score.to(enc_dtype)
        
        nn.init.normal_(self.head_score.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head_score.bias)

    def get_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward_score(self, input_ids, attention_mask):
        cls = self.get_cls(input_ids, attention_mask)
        cls = cls.to(self.head_score.weight.dtype)
        return self.head_score(cls)

class MixedABBatchSampler:
    def __init__(self, idx_a: List[int], idx_b: List[int], batch_size: int, seed: int = 42, strategy: str = "deterministic_concat"):
        self.idx_a = list(idx_a)
        self.idx_b = list(idx_b)
        self.bs = int(batch_size)
        self.half = self.bs // 2
        if self.half <= 0 or self.bs % 2 != 0:
            raise ValueError("batch_size must be even and >= 2")
        self.seed = int(seed)
        self.epoch = 0
        self.strategy = str(strategy)

    def __len__(self) -> int:
        return int(np.ceil(len(self.idx_a) / self.half))

    def __iter__(self) -> Iterator[List[int]]:
        self.epoch += 1

        a = np.array(self.idx_a, dtype=np.int64)
        b = np.array(self.idx_b, dtype=np.int64)

        if len(a) < self.half or len(b) == 0:
            return

        if self.strategy == "random_half":
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(a)
            rng.shuffle(b)
            b_ptr = 0
            b_len = len(b)
            for start in range(0, len(a), self.half):
                a_batch = a[start:start + self.half].tolist()
                if len(a_batch) < self.half:
                    break
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
            return

        if self.strategy == "deterministic_concat":
            b_ptr = 0
            b_len = len(b)
            for start in range(0, len(a), self.half):
                a_batch = a[start:start + self.half].tolist()
                if len(a_batch) < self.half:
                    break
                if b_ptr + self.half <= b_len:
                    b_batch = b[b_ptr:b_ptr + self.half].tolist()
                    b_ptr += self.half
                else:
                    tail = b[b_ptr:].tolist()
                    need = self.half - len(tail)
                    b_batch = tail + b[:need].tolist()
                    b_ptr = need
                yield a_batch + b_batch
            return

        raise ValueError(f"Unknown mix_strategy: {self.strategy}")

@torch.no_grad()
def evaluate(model: SarcasmModel, dlA: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, probs = [], []
    for batch in dlA:
        if "input_ids_a" not in batch:
            continue
        ids = batch["input_ids_a"].to(device)
        mask = batch["attention_mask_a"].to(device)
        labels = batch["label_a"].cpu().numpy()
        logits = model.forward_score(ids, mask).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(labels)
        probs.append(prob)

    f1_fixed = 0.0
    best_th = 0.5

    if ys:
        ys = np.concatenate(ys)
        probs = np.concatenate(probs)
        preds_fixed = (probs >= 0.5).astype(int)
        tp = np.sum((ys == 1) & (preds_fixed == 1))
        fp = np.sum((ys == 0) & (preds_fixed == 1))
        fn = np.sum((ys == 1) & (preds_fixed == 0))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1_fixed = 2 * (prec * rec) / max(1e-9, prec + rec)
        precisions, recalls, thresholds = precision_recall_curve(ys, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        f1_scores = np.nan_to_num(f1_scores)
        if len(f1_scores) > 0:
            idx = int(np.argmax(f1_scores))
            best_th = float(thresholds[idx]) if idx < len(thresholds) else 0.5

    return {"A_f1": float(f1_fixed), "score": float(f1_fixed), "best_th": float(best_th)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = pick_device(cfg["training"].get("device", "cuda"))

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    last_out_dir = out_dir / "last_checkpoint"
    last_out_dir.mkdir(parents=True, exist_ok=True)

    model_id = cfg["model"]["model_id"]
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    mix_strategy = str(cfg["training"].get("mix_strategy", "deterministic_concat"))

    lr_encoder = float(cfg["training"].get("lr_encoder", 2e-5))
    lr_head = float(cfg["training"].get("lr_head", 2e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 0.01))
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.1))

    lambda_a = float(cfg["training"].get("lambda_a", 1.0))
    lambda_b = float(cfg["training"].get("lambda_b", 0.5))
    margin = float(cfg["training"].get("margin", 1.0))

    pos_weight_val = cfg["training"].get("pos_weight", None)
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    qcfg = cfg.get("quantization", {}) or {}
    torch_dtype = parse_torch_dtype(qcfg.get("torch_dtype", None))
    if device.type != "cuda":
        use_amp = False
        if torch_dtype in (torch.float16, torch.bfloat16):
            torch_dtype = torch.float32

    tokenized_dir = cfg["data"].get("tokenized_dir", "../data/tokenized/o2")
    ds = load_from_disk(tokenized_dir)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    train_tasks = train_ds["task"]
    idx_a = [i for i, t in enumerate(train_tasks) if t == "A"]
    idx_b = [i for i, t in enumerate(train_tasks) if t == "B"]
    val_idx_a = [i for i, t in enumerate(val_ds["task"]) if t == "A"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    collator = SmartMixCollator(tokenizer)

    batch_sampler = MixedABBatchSampler(idx_a, idx_b, batch_size=batch_size, seed=seed, strategy=mix_strategy)

    num_workers = int(cfg["training"].get("num_workers", 2))
    dl_train = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    dlA_val = DataLoader(
        val_ds.select(val_idx_a),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    model = SarcasmModel(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
    ).to(device)

    enc_dtype = next(model.encoder.parameters()).dtype
    model.head_score = model.head_score.to(enc_dtype)

    param_groups = [
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": list(model.head_score.parameters()), "lr": lr_head, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(dl_train) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if pos_weight_val is not None:
        pos_weight_tensor = torch.tensor([float(pos_weight_val)], device=device)
        loss_fn_a = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        loss_fn_a = nn.BCEWithLogitsLoss()

    loss_fn_b = nn.MarginRankingLoss(margin=margin)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    loss = torch.tensor(0.0, device=device)

                    if "input_ids_a" in batch:
                        ids = batch["input_ids_a"].to(device)
                        mask = batch["attention_mask_a"].to(device)
                        y = batch["label_a"].to(device).unsqueeze(-1)
                        logits = model.forward_score(ids, mask)
                        loss = loss + lambda_a * loss_fn_a(logits, y)

                    if "input_ids_b_sarc" in batch:
                        ids_s = batch["input_ids_b_sarc"].to(device)
                        mask_s = batch["mask_b_sarc"].to(device)
                        ids_r = batch["input_ids_b_reph"].to(device)
                        mask_r = batch["mask_b_reph"].to(device)

                        score_s = model.forward_score(ids_s, mask_s).view(-1)
                        score_r = model.forward_score(ids_r, mask_r).view(-1)
                        target = torch.ones(score_s.size(0), device=device)
                        loss = loss + lambda_b * loss_fn_b(score_s, score_r, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                loss = torch.tensor(0.0, device=device)

                if "input_ids_a" in batch:
                    ids = batch["input_ids_a"].to(device)
                    mask = batch["attention_mask_a"].to(device)
                    y = batch["label_a"].to(device).unsqueeze(-1)
                    logits = model.forward_score(ids, mask)
                    loss = loss + lambda_a * loss_fn_a(logits, y)

                if "input_ids_b_sarc" in batch:
                    ids_s = batch["input_ids_b_sarc"].to(device)
                    mask_s = batch["mask_b_sarc"].to(device)
                    ids_r = batch["input_ids_b_reph"].to(device)
                    mask_r = batch["mask_b_reph"].to(device)

                    score_s = model.forward_score(ids_s, mask_s).view(-1)
                    score_r = model.forward_score(ids_r, mask_r).view(-1)
                    target = torch.ones(score_s.size(0), device=device)
                    loss = loss + lambda_b * loss_fn_b(score_s, score_r, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

        metrics = evaluate(model, dlA_val, device)
        avg_loss = total_loss / max(1, len(dl_train))

        print(f"\nEpoch {epoch}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  A_f1 (Fixed 0.5): {metrics['A_f1']:.4f} (Best Th Potential: {metrics['best_th']:.2f})")

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            model.encoder.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            torch.save(model.head_score.state_dict(), out_dir / "head_score.pt")
            (out_dir / "metrics_best.txt").write_text(str(metrics), encoding="utf-8")
            (out_dir / "train_mode.txt").write_text("full", encoding="utf-8")
            print(f"  Saved BEST to {out_dir}")

        model.encoder.save_pretrained(last_out_dir)
        tokenizer.save_pretrained(last_out_dir)
        torch.save(model.head_score.state_dict(), last_out_dir / "head_score.pt")
        (last_out_dir / "train_mode.txt").write_text("full", encoding="utf-8")
        print(f"  Saved LAST to {last_out_dir}")

    print(f"\nDone! Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()