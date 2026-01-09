# Report Valutazione Sarcasmo

**Modello:** `llama-3.3-70b-versatile`
**File Analizzato:** `isarcasmeval_test_predictions_llama-3.3-70b-versatile.jsonl`
**Totale Esempi Validi:** 1400
**Errori API/Parsing (Ignorati):** 0

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.4736** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.3472** | Media armonica tra Precision e Recall |
| **Precision** | 0.2110 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.9800 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **467** (True Negative) | **733** (False Positive) |
| **Reale: Sarcastico (1)** | **4** (False Negative) | **196** (True Positive) |

---

## 3. Report Dettagliato

```text
               precision    recall  f1-score   support

Non Sarcastic       0.99      0.39      0.56      1200
    Sarcastic       0.21      0.98      0.35       200

     accuracy                           0.47      1400
    macro avg       0.60      0.68      0.45      1400
 weighted avg       0.88      0.47      0.53      1400

