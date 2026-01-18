# Report Valutazione Sarcasmo (Phi-3 LoRA)

**Adapter:** `../results/runs/phi3_o1/step_400`  
**Base model:** `microsoft/Phi-3-mini-4k-instruct`  
**Dataset:** `reddit_sample_20k.jsonl`  
**Batch size:** `8`  
**Max prompt tokens:** `1024`  
**Label scoring:** `score = logP("A") - logP("B")`  
**Threshold (score=logP(A)-logP(B)):** **0.000**  
**Predicted positive rate:** **0.3181**

---

## Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore |
| :--- | :--- |
| **Accuracy** | **0.6003** |
| **F1-Score** | **0.5125** |
| **Precision** | 0.6605 |
| **Recall** | 0.4187 |

---

## Confusion Matrix

`Rows=True [0,1], Cols=Pred [0,1]`

- TN=7804  FP=2160
- FN=5834  TP=4202

---

## Report Dettagliato

```text
                precision    recall  f1-score   support

Non Sarcastic        0.57      0.78      0.66      9964
Sarcastic            0.66      0.42      0.51     10036

accuracy                             0.60     20000
macro avg            0.62      0.60      0.59     20000
weighted avg         0.62      0.60      0.59     20000
```