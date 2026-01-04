# cpu_cultural_enrich.py – switched to MiniCPM-V-2.6 for better & faster VQA + deep_translator
import os
os.environ["HF_HOME"] = r"D:\HF_CACHE"
os.environ["TRANSFORMERS_CACHE"] = r"D:\HF_CACHE\transformers"
os.environ["HF_HUB_CACHE"] = r"D:\HF_CACHE\hub"

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from deep_translator import GoogleTranslator  # ← New, reliable translator
import time

# Global translator instance (reused for efficiency)
translator = GoogleTranslator(source='en')  # source is always English

LANG_MAP = {
    "English": "en", "Hindi": "hi", "Urdu": "ur",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml",
    "Bengali": "bn", "Assamese": "as", "Odia": "or", "Manipuri": "mni-Mtei",
    "Bodo": "brx", "Marathi": "mr", "Gujarati": "gu", "Rajasthani": "raj",
    "Sindhi": "sd", "Punjabi": "pa", "Kashmiri": "ks", "Nepali": "ne",
    "Santali": "sat", "Maithili": "mai", "Dogri": "doi",
    "Konkani": "gom", "Sanskrit": "sa"
}

DEVICE = "cpu"
torch.set_num_threads(4)

# Florence-2 globals (base caption)
_florence_processor = None
_florence_model = None

# MiniCPM-V-2.6 globals (smarter & faster Q&A)
_minicpm_processor = None
_minicpm_model = None


def load_florence():
    global _florence_processor, _florence_model
    if _florence_model is None:
        print("[INFO] Loading Florence-2-base (lazy)")
        _florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
    return _florence_processor, _florence_model


def load_minicpm():
    global _minicpm_processor, _minicpm_model
    if _minicpm_model is None:
        print("[INFO] Loading MiniCPM-V-2.6 (lazy, float16)")
        _minicpm_processor = AutoProcessor.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        _minicpm_model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"
        ).eval()
    return _minicpm_processor, _minicpm_model


def generate_base_caption(path):
    processor, model = load_florence()
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    inputs = processor("<DETAILED_CAPTION>", img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, num_beams=3, do_sample=False)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def translate_to_language(text: str, lang: str) -> str:
    """
    Translate English text to target language using deep_translator (stable fork).
    Falls back to original text if translation fails.
    """
    if not text or lang == "English":
        return text

    target_code = LANG_MAP.get(lang, "en")
    if target_code == "en":
        return text

    try:
        translator.target = target_code
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"[WARN] Translation failed ({lang}): {e}")
        return text


def generate_vqa_answer(image_path: str, question: str):
    processor, model = load_minicpm()

    image = Image.open(image_path).convert("RGB")

    # MiniCPM-V chat-style interface (clean and fast)
    msgs = [{'role': 'user', 'content': question}]

    with torch.no_grad():
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=processor.tokenizer,
            sampling=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=120,
            repetition_penalty=1.05
        )

    answer = res.strip().capitalize()
    if answer and not answer.endswith(('.', '!', '?')):
        answer += '.'

    return answer