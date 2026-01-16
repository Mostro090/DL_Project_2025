# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **0.00**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8843** |
| **F1-Score** | **0.5781** |
| **Precision** | 0.6033 |
| **Recall** | 0.5550 |

---

## Matrice di Confusione

```text
[[1127  73]
 [89  111]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.93      0.94      0.93      1200
Sarcastic            0.60      0.56      0.58       200

accuracy                             0.88      1400
macro avg            0.77      0.75      0.76      1400
weighted avg         0.88      0.88      0.88      1400
```