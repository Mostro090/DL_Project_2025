# Report Valutazione Sarcasmo (Siamese Model)

**Modello:** `modernbert_o2`  
**File Analizzato:** `o2:test (thr=0.27)`  
**Totale Esempi Validi:** 1400

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.5500** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.2905** | Media armonica tra Precision e Recall |
| **Precision** | 0.1875 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.6450 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **641** (True Negative) | **559** (False Positive) |
| **Reale: Sarcastico (1)** | **71** (False Negative) | **129** (True Positive) |

---

## 3. Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.90      0.53      0.67      1200
Sarcastic            0.19      0.65      0.29       200

accuracy                             0.55      1400
macro avg            0.54      0.59      0.48      1400
weighted avg         0.80      0.55      0.62      1400
```