# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Modello Adapter:** `step_400`  
**File Analizzato:** `phi3_dataset:test`  
**Soglia (Logits A-B):** **0.00**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.7836** |
| **F1-Score** | **0.4992** |
| **Precision** | 0.3728 |
| **Recall** | 0.7550 |

---

## Matrice di Confusione

```text
[[946  254]
 [49  151]]
```

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.95      0.79      0.86      1200
Sarcastic            0.37      0.76      0.50       200

accuracy                             0.78      1400
macro avg            0.66      0.77      0.68      1400
weighted avg         0.87      0.78      0.81      1400
```

## Nota Bene
L'ablazione sarebbe anche sul pos_weight, abbiamo mantenuto l'idea rispetto al training di BERT. In questo caso, l'altra parte dell'ablazione coincide con phi3_o1, ovvero il best model globale.