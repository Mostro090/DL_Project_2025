# Report Valutazione Sarcasmo

**Modello:** `modernbert_ac_lora_mixbatch`  
**File Analizzato:** `tokenized:test (thr=0.31)`  
**Totale Esempi Validi:** 1400

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.1779** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.2492** | Media armonica tra Precision e Recall |
| **Precision** | 0.1433 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.9550 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **58** (True Negative) | **1142** (False Positive) |
| **Reale: Sarcastico (1)** | **9** (False Negative) | **191** (True Positive) |

---

## 3. Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.87      0.05      0.09      1200
Sarcastic            0.14      0.95      0.25       200

accuracy                             0.18      1400
macro avg            0.50      0.50      0.17      1400
weighted avg         0.76      0.18      0.11      1400
```