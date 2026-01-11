# Report Analisi Parametri LoRA: ModernBERT

**Modello Base:** `answerdotai/ModernBERT-base`
**Target Modules:** `["Wqkv", "Wo"]`
**LoRA Rank (r):** 8

---

## Statistiche Conteggio

| Metrica | Valore |
| :--- | :--- |
| **Totale Parametri Modello** | `150,163,200` |
| **Parametri Addestrabili (LoRA)** | `1,148,928` |
| **Parametri Congelati (Base)** | `149,014,272` |
| **% Parametri Addestrabili** | **`0.7651%`** |

## Dettagli Architettonici

In ModernBERT, l'attenzione è ottimizzata:
1. **`Wqkv`**: È una proiezione "fused" che gestisce Query, Key e Value in un unico layer denso per massimizzare l'efficienza della GPU.
2. **`Wo`**: È la proiezione di output standard dell'attenzione.

Allenare solo questi moduli con LoRA permette di adattare il meccanismo di attenzione mantenendo intatto il "knowledge" contenuto nei feed-forward networks (MLP) e negli embeddings.
