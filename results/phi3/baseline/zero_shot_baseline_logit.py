import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

IGNORE_INDEX = -100

def extract_prompt_ids_from_columns(input_ids_col, labels_col):
    prompts = []
    for ids, labs in zip(input_ids_col, labels_col):
        prompt_ids = [tid for tid, lab in zip(ids, labs) if lab == IGNORE_INDEX]
        prompts.append(prompt_ids)
    return prompts

@torch.no_grad()
def score_batch_next_token(model, tokenizer, batch_prompt_ids):
    device = model.device
    max_len = max(len(x) for x in batch_prompt_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = []
    attention_mask = []
    lengths = []

    for ids in batch_prompt_ids:
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

    A_ids = tokenizer.encode("A", add_special_tokens=False)
    B_ids = tokenizer.encode("B", add_special_tokens=False)
    
    A_id, B_id = A_ids[0], B_ids[0]
    scores = logp[:, A_id] - logp[:, B_id]
    return scores.detach().cpu()

def evaluate_zero_shot(
    dataset_dir: str,
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    split: str = "test",
    batch_size: int = 16,
    threshold: float = 0.0,
):
    ds = load_from_disk(dataset_dir)
    if split not in ds:
        raise ValueError(f"Split '{split}' not found.")

    dsplit = ds[split]
    if "task" in dsplit.column_names:
        dsplit = dsplit.filter(lambda x: x["task"] == "A")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    y_true, y_pred, all_scores = [], [], []
    n = len(dsplit)

    for i in tqdm(range(0, n, batch_size), desc="Evaluating batches"):
        batch = dsplit[i:i + batch_size]

        prompt_ids_batch = extract_prompt_ids_from_columns(
            batch["input_ids"],
            batch["labels"],
        )

        scores = score_batch_next_token(model, tokenizer, prompt_ids_batch)
        preds = (scores.float().numpy() > threshold).astype(int).tolist()

        gold = batch["gold_label"] if "gold_label" in batch else batch["label"]

        y_true.extend(list(gold))
        y_pred.extend(preds)
        all_scores.extend(scores.tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_report = classification_report(y_true, y_pred, digits=4)

    print(f"\n=== Report Final (Zero-Shot) ===")
    print(class_report)
    print("\nConfusion Matrix:")
    print(cm)
    
    with open("phi3_baseline_logit.md", "w", encoding="utf-8") as f:
        f.write(f"# Baseline Results\n\n```\n{class_report}\n```\n\n```\n{cm}\n```")

if __name__ == "__main__":
    dataset_dir = "../../../data/phi3_dataset"
    evaluate_zero_shot(
        dataset_dir=dataset_dir,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        split="test",
        batch_size=16,
        threshold=1.5 
    )