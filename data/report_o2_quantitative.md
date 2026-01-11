# Report Dataset: Strategia di Rincaro (Boosting)
**Dataset:** `o2_multitask` | **Totale Righe:** 5647

## 1. Situazione Iniziale (Solo Task A)
Il Task A riflette la distribuzione naturale delle classi nel dataset.
| Classe       | Conteggio | Percentuale | Note              |
| ------------ | --------- | ----------- | ----------------- |
| Negative (0) | 3800      | 78.1%       | Dominante         |
| Positive (1) | 1067      | 21.9%       | Target (Sarcasmo) |

> **Ratio Iniziale:** 1 Positivo ogni **3.6** Negativi.

## 2. Il Meccanismo di Rincaro (Task B)
Il Task B aggiunge esempi contrastivi *(Sarcasmo vs Parafrasi)* per aumentare l'esposizione del modello ai concetti positivi senza alterare le etichette di verità.
| Fonte Segnale               | Volume   | Ruolo                                   |
| --------------------------- | -------- | --------------------------------------- |
| Task A (Positivi)           | 1067     | Ground Truth (Classificazione)          |
| Task B (Boost)              | 780      | Supporto Semantico (Ranking)            |
| **Totale Segnali Positivi** | **1847** | Volume totale di apprendimento positivo |

## 3. Efficacia del Bilanciamento
Confronto tra lo sbilanciamento percepito dal modello prima e dopo l'aggiunta del Task B.
| Scenario             | Negativi vs Positivi | Ratio (Neg:Pos) | Note        |
| -------------------- | -------------------- | --------------- | ----------- |
| Solo Classificazione | 3800 vs 1067         | 3.56 : 1        | Sbilanciato |
| Multitask (Boosted)  | 3800 vs 1847         | **2.06 : 1**    | Mitigato    |

### Conclusione Quantitativa
- L'introduzione del Task Ausiliario aumenta il volume di informazioni positive del **+73.1%**.
- Lo sbilanciamento effettivo viene ridotto del **42.2%** (da 3.6:1 a 2.1:1).
- Questo permette al modello di vedere il concetto 'Sarcasmo' molto più frequentemente durante il training, compensando la scarsità di etichette.