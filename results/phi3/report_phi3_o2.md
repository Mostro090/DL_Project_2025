# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8971** |
| **F1-Score** | **0.5740** |
| **Precision** | 0.7029 |
| **Recall** | 0.4850 |

---

## Matrice di Confusione

```text
[[1159  41]
 [103  97]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.92      0.97      0.94      1200
Sarcastic            0.70      0.48      0.57       200

accuracy                             0.90      1400
macro avg            0.81      0.73      0.76      1400
weighted avg         0.89      0.90      0.89      1400
```