# Report Valutazione Sarcasmo (Phi-3)

**Adapter:** `../results/runs/phi3_contrastive_ano/step_200`  
**Base model:** `microsoft/Phi-3-mini-4k-instruct`  
**Dataset:** `reddit_sample_20k.jsonl`  
**Batch size:** `8`  
**Max prompt tokens:** `1024`  
**Label scoring:** `score = logP("A") - logP("B")`  
**Threshold (score=logP(A)-logP(B)):** **0.000** 
**Predicted positive rate:** **0.6581**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.6182** |
| **F1-Score** | **0.6709** |
| **Precision** | 0.5912 |
| **Recall** | 0.7753 |

---

## Confusion Matrix

`Rows=True [0,1], Cols=Pred [0,1]`

- TN=4584  FP=5380
- FN=2255  TP=7781

---

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.67      0.46      0.55      9964
Sarcastic            0.59      0.78      0.67     10036

accuracy                             0.62     20000
macro avg            0.63      0.62      0.61     20000
weighted avg         0.63      0.62      0.61     20000
```