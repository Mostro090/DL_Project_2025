# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_200`  
**File Analizzato:** `phi3_dataset_contrastive:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8914** |
| **F1-Score** | **0.6082** |
| **Precision** | 0.6277 |
| **Recall** | 0.5900 |

---

## Matrice di Confusione

```text
[[1130  70]
 [82  118]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.93      0.94      0.94      1200
Sarcastic            0.63      0.59      0.61       200

accuracy                             0.89      1400
macro avg            0.78      0.77      0.77      1400
weighted avg         0.89      0.89      0.89      1400
```