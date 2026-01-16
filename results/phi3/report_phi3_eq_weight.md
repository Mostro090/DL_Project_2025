# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **0.25**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8014** |
| **F1-Score** | **0.5106** |
| **Precision** | 0.3940 |
| **Recall** | 0.7250 |

---

## Matrice di Confusione

```text
[[977  223]
 [55  145]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.95      0.81      0.88      1200
Sarcastic            0.39      0.72      0.51       200

accuracy                             0.80      1400
macro avg            0.67      0.77      0.69      1400
weighted avg         0.87      0.80      0.82      1400
```