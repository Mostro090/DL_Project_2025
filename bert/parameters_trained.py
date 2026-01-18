import torch
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
import os

def generate_modernbert_lora_report():
    print("--- Caricamento ModernBERT-base ---")
    model_id = "answerdotai/ModernBERT-base"
    
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return

    print("--- Applicazione Configurazione LoRA ---")
    target_modules = ["Wqkv", "Wo"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = 0
    all_params = 0
    
    for _, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    if all_params > 0:
        percentage = (trainable_params / all_params) * 100
    else:
        percentage = 0

    print(f"Calcolo completato.")
    print(f"Parametri addestrabili: {trainable_params:,}")
    print(f"Totale parametri: {all_params:,}")

    md_content = f"""# Report Analisi Parametri LoRA: ModernBERT

**Modello Base:** `{model_id}`
**Target Modules:** `{target_modules}`
**LoRA Rank (r):** {lora_config.r}

---

## Statistiche Conteggio

| Metrica | Valore |
| :--- | :--- |
| **Totale Parametri Modello** | `{all_params:,}` |
| **Parametri Addestrabili (LoRA)** | `{trainable_params:,}` |
| **Parametri Congelati (Base)** | `{all_params - trainable_params:,}` |
| **% Parametri Addestrabili** | **`{percentage:.4f}%`** |

## Dettagli Architettonici

In ModernBERT, l'attenzione è ottimizzata:
1. **`Wqkv`**: È una proiezione "fused" che gestisce Query, Key e Value in un unico layer denso per massimizzare l'efficienza della GPU.
2. **`Wo`**: È la proiezione di output standard dell'attenzione.

Allenare solo questi moduli con LoRA permette di adattare il meccanismo di attenzione mantenendo intatto il "knowledge" contenuto nei feed-forward networks (MLP) e negli embeddings.
"""

    output_file = "report_lora.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Report generato con successo: {output_file}")

if __name__ == "__main__":
    generate_modernbert_lora_report()