"# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8943** |
| **F1-Score** | **0.5747** |
| **Precision** | 0.6757 |
| **Recall** | 0.5000 |

---

## Matrice di Confusione

```text
[[1152  48]
 [100  100]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.92      0.96      0.94      1200
Sarcastic            0.68      0.50      0.57       200

accuracy                             0.89      1400
macro avg            0.80      0.73      0.76      1400
weighted avg         0.89      0.89      0.89      1400
```"