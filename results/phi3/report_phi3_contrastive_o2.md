# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset_contrastive:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8493** |
| **F1-Score** | **0.5613** |
| **Precision** | 0.4804 |
| **Recall** | 0.6750 |

---

## Matrice di Confusione

```text
[[1054  146]
 [65  135]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.94      0.88      0.91      1200
Sarcastic            0.48      0.68      0.56       200

accuracy                             0.85      1400
macro avg            0.71      0.78      0.74      1400
weighted avg         0.88      0.85      0.86      1400
```