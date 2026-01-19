# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8850** |
| **F1-Score** | **0.5818** |
| **Precision** | 0.6054 |
| **Recall** | 0.5600 |

---

## Matrice di Confusione

```text
[[1127  73]
 [88  112]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.93      0.94      0.93      1200
Sarcastic            0.61      0.56      0.58       200

accuracy                             0.89      1400
macro avg            0.77      0.75      0.76      1400
weighted avg         0.88      0.89      0.88      1400
```