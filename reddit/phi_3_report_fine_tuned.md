# Report Valutazione Sarcasmo (Phi-3)

**Adapter:** `None (Base Model)`  
**Base model:** `microsoft/Phi-3-mini-4k-instruct`  
**Dataset:** `reddit_sample_20k.jsonl`  
**Batch size:** `8`  
**Max prompt tokens:** `1024`  
**Label scoring:** `score = logP("A") - logP("B")`  
**Threshold (score=logP(A)-logP(B)):** **0.000** 
**Predicted positive rate:** **0.3996**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.5918** |
| **F1-Score** | **0.5472** |
| **Precision** | 0.6172 |
| **Recall** | 0.4914 |

---

## Confusion Matrix

`Rows=True [0,1], Cols=Pred [0,1]`

- TN=6905  FP=3059
- FN=5104  TP=4932

---

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.57      0.69      0.63      9964
Sarcastic            0.62      0.49      0.55     10036

accuracy                             0.59     20000
macro avg            0.60      0.59      0.59     20000
weighted avg         0.60      0.59      0.59     20000
```