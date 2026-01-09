from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Iterator, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import precision_recall_curve

IGNORE_IDX = -100

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- COLLATOR CUSTOM PER GESTIRE BATCH MISTI ---
class SmartMixCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Separa gli esempi Task A da quelli Task B e li padda separatamente.
        Ritorna un dizionario con chiavi separate per i due task.
        """
        batch_a = [f for f in features if f["task"] == "A"]
        batch_b = [f for f in features if f["task"] == "B"]

        out = {}

        # --- Process Task A ---
        if batch_a:
            input_ids = [f["input_ids"] for f in batch_a]
            # Usa il pad di tokenizer
            padded = self.tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            )
            out["input_ids_a"] = padded["input_ids"]
            out["attention_mask_a"] = padded["attention_mask"]
            out["label_a"] = torch.tensor([f["label_a"] for f in batch_a], dtype=torch.float)

        # --- Process Task B (Siamese) ---
        if batch_b:
            # 1. Sarcastic branch
            ids_sarc = [f["input_ids_sarc"] for f in batch_b]
            pad_sarc = self.tokenizer.pad(
                {"input_ids": ids_sarc}, padding=True, return_tensors="pt"
            )
            out["input_ids_b_sarc"] = pad_sarc["input_ids"]
            out["mask_b_sarc"] = pad_sarc["attention_mask"]

            # 2. Rephrase branch
            ids_reph = [f["input_ids_reph"] for f in batch_b]
            pad_reph = self.tokenizer.pad(
                {"input_ids": ids_reph}, padding=True, return_tensors="pt"
            )
            out["input_ids_b_reph"] = pad_reph["input_ids"]
            out["mask_b_reph"] = pad_reph["attention_mask"]
            
            # Label B (utile solo per metriche legacy, non per la loss)
            out["label_b"] = torch.tensor([f["label_b"] for f in batch_b], dtype=torch.long)

        return out


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
        
        # UNICA TESTA: Predice uno "score di sarcasmo"
        # Task A: sigmoid(score) -> probabilità
        # Task B: score(sarcastic) > score(rephrase)
        self.head_score = nn.Linear(hidden, 1)

    def get_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]
    
    def forward_score(self, input_ids, attention_mask):
        cls = self.get_cls(input_ids, attention_mask)
        return self.head_score(cls)


class MixedABBatchSampler:
    def __init__(self, idx_a: List[int], idx_b: List[int], batch_size: int, seed: int = 42):
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
                break # Drop last incomplete

            if b_len == 0: continue

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
def evaluate(model: SarcasmModel, dlA: DataLoader, dlB: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    # --- TASK A: Dynamic Threshold ---
    ys, probs = [], []
    for batch in dlA:
        # Nota: dlA restituisce batch processati dallo SmartCollator
        # Controlliamo se ci sono dati A (dovrebbero esserci sempre in dlA)
        if "input_ids_a" not in batch: continue
        
        ids = batch["input_ids_a"].to(device)
        mask = batch["attention_mask_a"].to(device)
        labels = batch["label_a"].cpu().numpy()

        logits = model.forward_score(ids, mask).squeeze(-1)
        prob = torch.sigmoid(logits).cpu().numpy()
        
        ys.append(labels)
        probs.append(prob)

    f1_a, best_th = 0.0, 0.5
    if ys:
        ys = np.concatenate(ys)
        probs = np.concatenate(probs)
        
        # Calcolo Soglia Ottimale
        precisions, recalls, thresholds = precision_recall_curve(ys, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        f1_scores = np.nan_to_num(f1_scores)
        if len(f1_scores) > 0:
            idx = np.argmax(f1_scores)
            best_f1 = f1_scores[idx]
            best_th = thresholds[idx] if idx < len(thresholds) else 0.5
            f1_a = float(best_f1)

    # --- TASK B: Pairwise Accuracy ---
    correct = 0
    total = 0
    for batch in dlB:
        if "input_ids_b_sarc" not in batch: continue

        ids_s = batch["input_ids_b_sarc"].to(device)
        mask_s = batch["mask_b_sarc"].to(device)
        ids_r = batch["input_ids_b_reph"].to(device)
        mask_r = batch["mask_b_reph"].to(device)

        score_s = model.forward_score(ids_s, mask_s)
        score_r = model.forward_score(ids_r, mask_r)

        # Predizione corretta se score(sarc) > score(reph)
        preds = (score_s > score_r).float()
        correct += preds.sum().item()
        total += preds.size(0)

    acc_b = correct / total if total > 0 else 0.0

    return {
        "A_f1": f1_a, 
        "B_acc": acc_b, 
        "score": f1_a + 0.2 * acc_b,
        "best_th": float(best_th)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))

    device_str = cfg["training"].get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # ATTENZIONE: Assicurati che punti alla nuova cartella tokenizzata
    tokenized_dir = cfg["data"].get("tokenized_dir", "../data/tokenized/tokenized_siamese")
    
    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = cfg["model"]["model_id"]
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    lambda_a = float(cfg["training"].get("lambda_a", 1.0))
    lambda_b = float(cfg["training"].get("lambda_b", 0.5)) 
    margin = float(cfg["training"].get("margin", 1.0)) # Margine per Ranking Loss

    ds = load_from_disk(tokenized_dir)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    # Separazione indici per Sampler
    train_tasks = train_ds["task"]
    idx_a = [i for i, t in enumerate(train_tasks) if t == "A"]
    idx_b = [i for i, t in enumerate(train_tasks) if t == "B"]
    
    val_idx_a = [i for i, t in enumerate(val_ds["task"]) if t == "A"]
    val_idx_b = [i for i, t in enumerate(val_ds["task"]) if t == "B"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    collator = SmartMixCollator(tokenizer)

    batch_sampler = MixedABBatchSampler(idx_a, idx_b, batch_size=batch_size, seed=int(cfg.get("seed", 42)))
    
    dl_train = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )

    # DataLoader separati per validation
    dlA_val = DataLoader(val_ds.select(val_idx_a), batch_size=batch_size, collate_fn=collator)
    dlB_val = DataLoader(val_ds.select(val_idx_b), batch_size=batch_size, collate_fn=collator)

    # Pos Weight per Task A (Sbilanciamento)
    y_a = np.array([x for x in train_ds.select(idx_a)["label_a"] if x != IGNORE_IDX])
    n_pos = (y_a == 1).sum()
    n_neg = (y_a == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    
    print(f"Pos Weight A: {pos_weight.item():.2f}")

    model = SarcasmModel(model_id=model_id, peft_cfg=cfg.get("peft", {})).to(device)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(len(dl_train)*0.06), len(dl_train)*epochs)

    # Loss Functions
    # Task A: Binary Cross Entropy
    loss_fn_a = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Task B: Margin Ranking Loss (Contrastive)
    loss_fn_b = nn.MarginRankingLoss(margin=margin)

    best_score = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            loss = torch.tensor(0.0, device=device)
            
            # --- TASK A STEP ---
            if "input_ids_a" in batch:
                ids = batch["input_ids_a"].to(device)
                mask = batch["attention_mask_a"].to(device)
                y = batch["label_a"].to(device).unsqueeze(-1) # (B, 1)
                
                logits = model.forward_score(ids, mask)
                loss_curr_a = loss_fn_a(logits, y)
                loss += lambda_a * loss_curr_a

            # --- TASK B STEP (Ranking) ---
            if "input_ids_b_sarc" in batch:
                ids_s = batch["input_ids_b_sarc"].to(device)
                mask_s = batch["mask_b_sarc"].to(device)
                ids_r = batch["input_ids_b_reph"].to(device)
                mask_r = batch["mask_b_reph"].to(device)
                
                # MODIFICA QUI: Aggiungi .view(-1) per appiattire i tensori a [Batch_Size]
                score_s = model.forward_score(ids_s, mask_s).view(-1)
                score_r = model.forward_score(ids_r, mask_r).view(-1)
                
                # Target=1 significa che input1 (score_s) deve essere > input2 (score_r)
                target = torch.ones(score_s.size(0), device=device)
                
                loss_curr_b = loss_fn_b(score_s, score_r, target)
                loss += lambda_b * loss_curr_b

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- EVALUATION ---
        metrics = evaluate(model, dlA_val, dlB_val, device)
        avg_loss = total_loss / max(1, len(dl_train))
        
        print(f"\nEpoch {epoch}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  A_f1: {metrics['A_f1']:.4f} (Th: {metrics['best_th']:.2f}) | B_acc: {metrics['B_acc']:.4f}")
        
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            model.encoder.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            torch.save(model.head_score.state_dict(), out_dir / "head_score.pt")
            (out_dir / "metrics_best.txt").write_text(str(metrics), encoding="utf-8")
            print(f"  Saved best to {out_dir}")

    print(f"\nDone! Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()