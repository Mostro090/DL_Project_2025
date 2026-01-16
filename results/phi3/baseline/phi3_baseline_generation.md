# Phi-3 Generative Baseline Report (Apples-to-Apples)

## Configuration
- **Type**: Generative (model.generate)
- **Model**: microsoft/Phi-3-mini-4k-instruct
- **Decoding**: Greedy (do_sample=False)

## Metrics
- **Precision (sarcastic=1)**: 0.2416
- **Recall    (sarcastic=1)**: 0.7900
- **F1        (sarcastic=1)**: 0.3700

## Confusion Matrix
(Rows = True [0,1], Cols = Pred [0,1])
[[704 496]
 [ 42 158]]

## Classification Report
              precision    recall  f1-score   support

           0     0.9437    0.5867    0.7235      1200
           1     0.2416    0.7900    0.3700       200

    accuracy                         0.6157      1400
   macro avg     0.5926    0.6883    0.5468      1400
weighted avg     0.8434    0.6157    0.6730      1400

