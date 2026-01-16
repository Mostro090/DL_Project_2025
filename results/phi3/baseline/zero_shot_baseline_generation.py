import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

IGNORE_INDEX = -100

def extract_prompts_and_gold(batch):
    prompts = []
    golds = []
    
    input_ids_col = batch["input_ids"]
    labels_col = batch["labels"]
    gold_col = batch["gold_label"] if "gold_label" in batch else batch["label"]

    for ids, labs, true_label in zip(input_ids_col, labels_col, gold_col):
        prompt_ids = [tid for tid, lab in zip(ids, labs) if lab == IGNORE_INDEX]
        prompts.append(prompt_ids)
        golds.append(true_label)
        
    return prompts, golds

def generate_batch(model, tokenizer, batch_prompt_ids):
    device = model.device
    
    max_len = max(len(x) for x in batch_prompt_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    input_ids = []
    attention_mask = []
    
    for ids in batch_prompt_ids:
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)

    outputs = model.generate(
        input_ids=input_tensor,
        attention_mask=mask_tensor,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=pad_id
    )

    generated_ids = outputs[:, input_tensor.shape[1]:]
    decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return decoded_texts

def evaluate_generative(
    dataset_dir: str,
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    split: str = "test",
    batch_size: int = 16,
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
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    y_true = []
    y_pred = []
    
    raw_responses = [] 

    n = len(dsplit)
    print(f"Starting Generative Evaluation on {n} samples...")

    for i in tqdm(range(0, n, batch_size), desc="Generating"):
        batch = dsplit[i:i + batch_size]
        
        prompt_ids, gold_labels = extract_prompts_and_gold(batch)
        
        responses = generate_batch(model, tokenizer, prompt_ids)
        
        batch_preds = []
        for r in responses:
            clean_r = r.strip()
            raw_responses.append(clean_r)
            
            if clean_r == "A":
                batch_preds.append(1) 
            elif clean_r == "B":
                batch_preds.append(0) 
            else:
                batch_preds.append(0) 

        y_true.extend(gold_labels)
        y_pred.extend(batch_preds)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_report = classification_report(y_true, y_pred, digits=4)

    invalid_count = len([x for x in raw_responses if x not in ["A", "B"]])
    if invalid_count > 0:
        print(f"\n[WARNING] Found {invalid_count} invalid responses (neither A nor B). Treated as 0.")
        print("Example invalid responses:", list(set([x for x in raw_responses if x not in ["A", "B"]]))[:5])

    report_content = f"""# Phi-3 Generative Baseline Report (Apples-to-Apples)

## Configuration
- **Type**: Generative (model.generate)
- **Model**: {model_name}
- **Decoding**: Greedy (do_sample=False)

## Metrics
- **Precision (sarcastic=1)**: {p:.4f}
- **Recall    (sarcastic=1)**: {r:.4f}
- **F1        (sarcastic=1)**: {f1:.4f}

## Confusion Matrix
(Rows = True [0,1], Cols = Pred [0,1])
{cm}

## Classification Report
{class_report}
"""

    print(report_content)

    with open("phi3_baseline_generation.md", "w", encoding="utf-8") as f:
        f.write(report_content)

if __name__ == "__main__":
    dataset_dir = "../../../data/phi3_dataset"
    evaluate_generative(
        dataset_dir=dataset_dir,
        model_name="microsoft/Phi-3-mini-4k-instruct",
        split="test",
        batch_size=8,
    )