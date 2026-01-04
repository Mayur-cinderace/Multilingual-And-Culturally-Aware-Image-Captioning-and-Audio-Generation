# translation_refiner.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

DEVICE = "cpu"
_tokenizer = None
_model = None

def load_translation_refiner():
    global _tokenizer, _model
    if _model is None:
        print("[INFO] Loading translation refiner (flan-t5-small)")
        _tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE).eval()
    return _tokenizer, _model


def naturalize_translation(text: str, language: str) -> str:
    """
    Makes translated text sound native.
    NEVER returns empty string.
    """

    text = text.strip()
    if not text:
        return text

    tok, model = load_translation_refiner()

    prompt = (
        f"You are a native speaker of {language}.\n"
        f"Rewrite the following sentence so it sounds natural and fluent.\n"
        f"Keep the meaning exactly the same.\n"
        f"Use one sentence only.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Natural sentence:"
    )

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            min_new_tokens=5,          # 🔴 important
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )

    refined = tok.decode(outputs[0], skip_special_tokens=True).strip()

    # 🔐 SAFETY FALLBACK
    if not refined or len(refined) < 3:
        return text

    return refined
