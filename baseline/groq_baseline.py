import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from groq import AsyncGroq  

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results/llm"

MODEL = "moonshotai/kimi-k2-instruct-0905"
SAFE_MODEL = MODEL.replace("/", "_")

INPUT_JSONL = str(DATA_DIR / "isarcasmeval_test.jsonl")
OUTPUT_JSONL = str(RESULTS_DIR / f"isarcasmeval_test_predictions_{SAFE_MODEL}.jsonl")

GROQ_API_KEY = ":)"

MAX_CONCURRENCY = 15 
MAX_RETRIES = 3

SYSTEM_MSG = (
    "You are a strict binary classifier for sarcasm detection.\n"
    "Given a short social media post, decide if it is sarcastic.\n"
    "Output ONLY a single character:\n"
    "1 = sarcastic\n"
    "0 = not sarcastic"
)

def parse_label(text: str) -> Optional[int]:
    if not text: return None
    if "1" in text: return 1
    if "0" in text: return 0
    return None

async def process_single_item(client: AsyncGroq, semaphore: asyncio.Semaphore, item: Dict):
    txt = item.get("text", "")
    
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": f"Text:\n{txt}\n\nAnswer (0 or 1):"},
                    ],
                    temperature=0.0,
                    max_completion_tokens=5,
                )
                
                raw = resp.choices[0].message.content
                pred = parse_label(raw)
                
                if pred is not None:
                    return {**item, "pred": pred, "raw": raw}
                
                return {**item, "pred": None, "raw": raw, "error": "parsing_failed"}

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {**item, "pred": None, "raw": None, "error": str(e)}
                
                await asyncio.sleep(1 * (2 ** attempt))

    return {**item, "pred": None, "error": "unknown"}

async def main():
    try:
        with open(INPUT_JSONL, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File non trovato: {INPUT_JSONL}")
        return

    print(f"Caricati {len(data)} esempi. Inizio elaborazione asincrona...")

    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [process_single_item(client, semaphore, item) for item in data]

    results = []
    total = len(tasks)
    
    for i, future in enumerate(asyncio.as_completed(tasks)):
        res = await future
        results.append(res)
        
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{total}] completati...")
                
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Finito! Salvato in {OUTPUT_JSONL}")

if __name__ == "__main__":
    asyncio.run(main())