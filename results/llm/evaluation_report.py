import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

MODEL = "moonshotai/kimi-k2-instruct-0905"
SAFE_MODEL = MODEL.replace("/", "_")
INPUT_FILE = f"isarcasmeval_test_predictions_{SAFE_MODEL}.jsonl" 
OUTPUT_MD = f"evaluation_report_{SAFE_MODEL}.md"

def load_predictions(path):
    y_true = []
    y_pred = []
    errors = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                
                gold = data.get("label") 
                pred = data.get("pred")

                if gold is not None and pred is not None:
                    y_true.append(int(gold))
                    y_pred.append(int(pred))
                else:
                    errors += 1
    except FileNotFoundError:
        print(f"Error: cant find {path}")
        return [], [], 0
                
    return y_true, y_pred, errors

def generate_markdown_report(y_true, y_pred, error_count):
    if not y_true: return "No data."

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0 

    md = f"""# Report Valutazione Sarcasmo

**Modello:** `{MODEL}`
**File Analizzato:** `{INPUT_FILE}`
**Totale Esempi Validi:** {len(y_true)}
**Errori API/Parsing (Ignorati):** {error_count}

---

## 1. Metriche Principali (Classe 'Sarcastic')

| Metrica | Valore | Descrizione |
| :--- | :--- | :--- |
| **Accuracy** | **{acc:.4f}** | Percentuale totale di risposte corrette |
| **F1-Score** | **{f1:.4f}** | Media armonica tra Precision e Recall |
| **Precision** | {p:.4f} | Quando predice "Sarcasmo", quanto spesso ha ragione? |
| **Recall** | {r:.4f} | Quanto sarcasmo reale riesce a trovare? |

---

## 2. Matrice di Confusione

| | **Predetto: Non Sarcastico (0)** | **Predetto: Sarcastico (1)** |
| :--- | :---: | :---: |
| **Reale: Non Sarcastico (0)** | **{tn}** (True Negative) | **{fp}** (False Positive) |
| **Reale: Sarcastico (1)** | **{fn}** (False Negative) | **{tp}** (True Positive) |

---

## 3. Report Dettagliato

```text
{classification_report(y_true, y_pred, target_names=['Non Sarcastic', 'Sarcastic'])}
"""
    return md

def main(): 
    Path(OUTPUT_MD).parent.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, errors = load_predictions(INPUT_FILE)
    report_md = generate_markdown_report(y_true, y_pred, errors)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report scritto in: {OUTPUT_MD}")
    print(f"Esempi validi: {len(y_true)} | Errori/righe ignorate: {errors}")

if __name__ == "__main__":
    main()
