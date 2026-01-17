import os
import re
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from groq import AsyncGroq

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = Path(__file__).resolve().parent

KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshotai/kimi-k2-instruct-0905")
LLAMA70B_MODEL = os.getenv("LLAMA70B_MODEL", "llama-3.3-70b-versatile")
PHI3_MODEL = os.getenv("PHI3_MODEL", "microsoft/Phi-3-mini-4k-instruct")

OUTPUT_MD = BASE_DIR / "check.md"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", ":)")

MAX_RETRIES = 3
REMOTE_MAX_TOKENS = 8
REMOTE_TEMPERATURE = 0.0

LOCAL_MAX_NEW_TOKENS = 12
LOCAL_TEMPERATURE = 0.0

SYSTEM_MSG = (
    "You are a strict binary classifier for sarcasm detection.\n"
    "Given a short social media post, decide if it is sarcastic.\n"
    "Output ONLY a single character:\n"
    "1 = sarcastic\n"
    "0 = not sarcastic"
)

EXAMPLES = [
    {
        "id": "helpful_short",
        "title": 'The Praise - SHORT (Passive-Aggressive Trap)',
        "text": "Very helpful."
    },
    {
        "id": "helpful_long",
        "title": 'The Praise - LONG (Sincere Context)',
        "text": "Very helpful. The tutorial explained exactly how to solve the error in just a few steps."
    },
    {
        "id": "impressed_short",
        "title": 'The Reaction - SHORT (Ambiguous)',
        "text": "Wow. Just wow."
    },
    {
        "id": "impressed_long",
        "title": 'The Reaction - LONG (Sincere)',
        "text": "Wow. Just wow. The view from the top of this mountain is absolutely breathtaking, I have no words."
    }
]

_LABEL_RE = re.compile(r"\b([01])\b")

def parse_label(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip()
    m = _LABEL_RE.search(s)
    if m:
        return int(m.group(1))
    s2 = re.sub(r"[^01]", "", s)
    if s2 in ("0", "1"):
        return int(s2)
    return None

def md_escape(s: str) -> str:
    return s.replace("```", "``\\`")

@dataclass
class ModelOutput:
    model_name: str
    model_id: str
    raw: str
    parsed: Optional[int]
    error: Optional[str] = None


async def groq_classify(
    client: AsyncGroq,
    model_id: str,
    text: str,
    max_retries: int = MAX_RETRIES
) -> ModelOutput:
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": f"Text:\n{text}\n\nAnswer (0 or 1):"},
                ],
                temperature=REMOTE_TEMPERATURE,
                max_completion_tokens=REMOTE_MAX_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
            pred = parse_label(raw)
            return ModelOutput(
                model_name="groq",
                model_id=model_id,
                raw=raw,
                parsed=pred,
                error=None if pred is not None else "parsing_failed",
            )
        except Exception as e:
            if attempt == max_retries - 1:
                return ModelOutput(
                    model_name="groq",
                    model_id=model_id,
                    raw="",
                    parsed=None,
                    error=str(e),
                )
            await asyncio.sleep(1 * (2 ** attempt))

    return ModelOutput(model_name="groq", model_id=model_id, raw="", parsed=None, error="unknown")


class Phi3Local:
    def __init__(self, model_id: str):
        self.model_id = model_id
        print(f"Loading local model: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        print("Local model loaded.")

    @torch.inference_mode()
    def classify(self, text: str) -> ModelOutput:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Text:\n{text}\n\nAnswer (0 or 1):"},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{SYSTEM_MSG}\n\nText:\n{text}\n\nAnswer (0 or 1):"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=False if LOCAL_TEMPERATURE == 0 else True,
            temperature=LOCAL_TEMPERATURE,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        raw = out.strip()
        pred = parse_label(raw)

        return ModelOutput(
            model_name="local",
            model_id=self.model_id,
            raw=raw,
            parsed=pred,
            error=None if pred is not None else "parsing_failed",
        )

def build_markdown(
    outputs: Dict[str, Dict[str, ModelOutput]],
    meta: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Sarcasm Detection Probe — Passive Aggressive Test")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- Kimi (Groq): `{meta['kimi_model']}`")
    lines.append(f"- Llama 70B (Groq): `{meta['llama_model']}`")
    lines.append(f"- Phi-3 (local): `{meta['phi3_model']}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")

    for ex in EXAMPLES:
        ex_id = ex["id"]
        lines.append(f"### {ex['title']}")
        lines.append("")
        lines.append("**Text**")
        lines.append("")
        lines.append("```text")
        lines.append(md_escape(ex["text"]))
        lines.append("```")
        lines.append("")
        lines.append("**Model responses**")
        lines.append("")
        lines.append("| Model | Model ID | Raw output | Parsed (0/1) | Error |")
        lines.append("|---|---|---|---:|---|")

        order = ["kimi", "phi3", "llama70b"]
        for key in order:
            mo = outputs[ex_id][key]
            raw_one_line = mo.raw.replace("\n", "\\n").strip()
            if len(raw_one_line) > 200:
                raw_one_line = raw_one_line[:200] + "…"
            lines.append(
                f"| {key} | `{mo.model_id}` | `{raw_one_line}` | {'' if mo.parsed is None else mo.parsed} | {mo.error or ''} |"
            )

        lines.append("")
        lines.append("**Raw (full)**")
        lines.append("")
        for key in order:
            mo = outputs[ex_id][key]
            lines.append(f"#### {key} — `{mo.model_id}`")
            lines.append("")
            lines.append("```text")
            lines.append(md_escape(mo.raw if mo.raw else ""))
            lines.append("```")
            if mo.error:
                lines.append(f"- Error: `{mo.error}`")
            lines.append("")

    return "\n".join(lines)


async def main():
    print("Initializing clients...")
    client = AsyncGroq(api_key=GROQ_API_KEY)
    phi3 = Phi3Local(PHI3_MODEL)

    outputs: Dict[str, Dict[str, ModelOutput]] = {}

    print(f"Running inference on {len(EXAMPLES)} examples...")
    for ex in EXAMPLES:
        ex_id = ex["id"]
        text = ex["text"]
        print(f"Processing: {ex_id}")
        outputs[ex_id] = {}

        kimi_task = asyncio.create_task(groq_classify(client, KIMI_MODEL, text))
        llama_task = asyncio.create_task(groq_classify(client, LLAMA70B_MODEL, text))

        phi_out = phi3.classify(text)

        kimi_out, llama_out = await asyncio.gather(kimi_task, llama_task)

        outputs[ex_id]["kimi"] = kimi_out
        outputs[ex_id]["llama70b"] = llama_out
        outputs[ex_id]["phi3"] = phi_out

    meta = {
        "kimi_model": KIMI_MODEL,
        "llama_model": LLAMA70B_MODEL,
        "phi3_model": PHI3_MODEL,
    }

    md = build_markdown(outputs, meta)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"Saved markdown to: {OUTPUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())