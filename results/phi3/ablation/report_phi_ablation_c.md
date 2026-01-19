# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **1.10**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8921** |
| **F1-Score** | **0.4664** |
| **Precision** | 0.7952 |
| **Recall** | 0.3300 |

---

## Matrice di Confusione

```text
[[1183  17]
 [134  66]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.90      0.99      0.94      1200
Sarcastic            0.80      0.33      0.47       200

accuracy                             0.89      1400
macro avg            0.85      0.66      0.70      1400
weighted avg         0.88      0.89      0.87      1400
```