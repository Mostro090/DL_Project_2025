# Report Valutazione Sarcasmo (Last Checkpoint)

**Modello:** `modernbert_o2`  
**File Analizzato:** `o2:test`  
**Soglia Utilizzata:** **0.50** (Fixed)

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **0.7929** | Percentuale totale di risposte corrette |
| **F1-Score** | **0.3439** | Media armonica tra Precision e Recall |
| **Precision** | 0.3140 | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | 0.3800 | Quanto sarcasmo reale riesce a trovare? |

---

## 3. Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.89      0.86      0.88      1200
Sarcastic            0.31      0.38      0.34       200

accuracy                             0.79      1400
macro avg            0.60      0.62      0.61      1400
weighted avg         0.81      0.79      0.80      1400
```