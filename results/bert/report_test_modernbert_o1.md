# Report Valutazione Sarcasmo

**Modello:** `modernbert_o1`  
**File Analizzato:** `o1:test (thr=0.50)`  
**Totale Esempi Validi:** 1400

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.1564** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.2454** | Media armonica tra Precision e Recall |
| **Precision** | 0.1407 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.9600 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **27** (True Negative) | **1173** (False Positive) |
| **Reale: Sarcastico (1)** | **8** (False Negative) | **192** (True Positive) |

---

## 3. Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.77      0.02      0.04      1200
Sarcastic            0.14      0.96      0.25       200

accuracy                             0.16      1400
macro avg            0.46      0.49      0.14      1400
weighted avg         0.68      0.16      0.07      1400
```