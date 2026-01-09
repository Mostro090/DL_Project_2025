# Report Valutazione Sarcasmo (Siamese Model)

**Modello:** `modernbert_o2`  
**File Analizzato:** `o2:test (thr=0.50)`  
**Totale Esempi Validi:** 1400

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.4943** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.2354** | Media armonica tra Precision e Recall |
| **Precision** | 0.1501 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.5450 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **583** (True Negative) | **617** (False Positive) |
| **Reale: Sarcastico (1)** | **91** (False Negative) | **109** (True Positive) |

---

## 3. Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.86      0.49      0.62      1200
Sarcastic            0.15      0.55      0.24       200

accuracy                             0.49      1400
macro avg            0.51      0.52      0.43      1400
weighted avg         0.76      0.49      0.57      1400
```