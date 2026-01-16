# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **0.25**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8193** |
| **F1-Score** | **0.5235** |
| **Precision** | 0.4199 |
| **Recall** | 0.6950 |

---

## Matrice di Confusione

```text
[[1008  192]
 [61  139]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.94      0.84      0.89      1200
Sarcastic            0.42      0.69      0.52       200

accuracy                             0.82      1400
macro avg            0.68      0.77      0.71      1400
weighted avg         0.87      0.82      0.84      1400
```