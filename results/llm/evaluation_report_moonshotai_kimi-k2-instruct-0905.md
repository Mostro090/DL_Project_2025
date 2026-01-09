# Report Valutazione Sarcasmo

**Modello:** `moonshotai/kimi-k2-instruct-0905`
**File Analizzato:** `isarcasmeval_test_predictions_moonshotai_kimi-k2-instruct-0905.jsonl`
**Totale Esempi Validi:** 1400
**Errori API/Parsing (Ignorati):** 0

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.5857** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.4008** | Media armonica tra Precision e Recall |
| **Precision** | 0.2526 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.9700 | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **626** (True Negative) | **574** (False Positive) |
| **Reale: Sarcastico (1)** | **6** (False Negative) | **194** (True Positive) |

---

## 3. Report Dettagliato

```text
               precision    recall  f1-score   support

Non Sarcastic       0.99      0.52      0.68      1200
    Sarcastic       0.25      0.97      0.40       200

     accuracy                           0.59      1400
    macro avg       0.62      0.75      0.54      1400
 weighted avg       0.89      0.59      0.64      1400

