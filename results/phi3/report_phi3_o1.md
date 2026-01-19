# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **0.00**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.8807** |
| **F1-Score** | **0.5617** |
| **Precision** | 0.5912 |
| **Recall** | 0.5350 |

---

## Matrice di Confusione

```text
[[1126  74]
 [93  107]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.92      0.94      0.93      1200
Sarcastic            0.59      0.54      0.56       200

accuracy                             0.88      1400
macro avg            0.76      0.74      0.75      1400
weighted avg         0.88      0.88      0.88      1400
```