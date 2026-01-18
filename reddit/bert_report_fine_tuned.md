# Report Valutazione Sarcasmo (ModernBERT LoRA)

**Model dir:** `../results/runs/modernbert_o2_second`  
**Base model:** `answerdotai/ModernBERT-base`  
**Dataset:** `reddit_sample_20k.jsonl`  
**Batch size:** `32`  
**Max length:** `256`  
**Threshold (sigmoid(logit) >= t):** **0.75**  
**Predicted positive rate:** **0.2671**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.5654** |
| **F1-Score** | **0.4348** |
| **Precision** | 0.6258 |
| **Recall** | 0.3331 |

---

## Confusion Matrix

`Rows=True [0,1], Cols=Pred [0,1]`

- TN=7965  FP=1999
- FN=6693  TP=3343

---

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.54      0.80      0.65      9964
Sarcastic            0.63      0.33      0.43     10036

accuracy                             0.57     20000
macro avg            0.58      0.57      0.54     20000
weighted avg         0.58      0.57      0.54     20000
```