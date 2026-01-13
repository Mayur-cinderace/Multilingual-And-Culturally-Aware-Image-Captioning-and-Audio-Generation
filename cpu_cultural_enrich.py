# cpu_cultural_enrich.py – Florence-2 caption + MiniCPM-V Q&A (HF only)
import os
os.environ["HF_HOME"] = r"D:\HF_CACHE"
os.environ["TRANSFORMERS_CACHE"] = r"D:\HF_CACHE\transformers"
os.environ["HF_HUB_CACHE"] = r"D:\HF_CACHE\hub"

import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from deep_translator import GoogleTranslator
import time

translator = GoogleTranslator(source='en')

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

_florence_processor = None
_florence_model = None

_minicpm_processor = None
_minicpm_model = None


def load_florence():
    global _florence_processor, _florence_model
    if _florence_model is None:
        print("[INFO] Loading Florence-2-base ...")
        _florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(DEVICE).eval()
    return _florence_processor, _florence_model


def load_minicpm():
    global _minicpm_processor, _minicpm_model
    if _minicpm_model is None:
        print("[INFO] Loading MiniCPM-V-2.6 ...")
        _minicpm_processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
        _minicpm_model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu"
        ).eval()
    return _minicpm_processor, _minicpm_model


def generate_base_caption(path: str) -> str:
    if not os.path.exists(path):
        return "Error: Image file not found."

    processor, model = load_florence()

    img_cv = cv2.imread(path)
    if img_cv is None:
        return "Error: Could not read image."

    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    task_prompt = "<DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=130,
            num_beams=3,
            do_sample=False
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption = generated_text.replace(task_prompt, "").replace("<", "").replace(">", "").strip()

    if not caption.endswith(('.', '!', '?')):
        caption += "."

    return caption


def translate_to_language(text: str, lang: str) -> str:
    if not text or lang == "English":
        return text

    target_code = LANG_MAP.get(lang, "en")
    if target_code == "en":
        return text

    try:
        translator.target = target_code
        translated = translator.translate(text)
        return translated or text
    except Exception as e:
        print(f"[Translation warning] {lang}: {e}")
        return text


def generate_vqa_answer(image_path: str, question: str) -> str:
    if not os.path.exists(image_path):
        return "Error: Image not found."

    processor, model = load_minicpm()
    image = Image.open(image_path).convert("RGB")

    msgs = [{'role': 'user', 'content': question}]

    try:
        with torch.no_grad():
            res = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=processor.tokenizer,
                sampling=True,
                temperature=0.75,
                top_p=0.9,
                max_tokens=140,
                repetition_penalty=1.05
            )
        answer = res.strip().capitalize()
        if answer and not answer.endswith(('.', '!', '?')):
            answer += "."
        return answer
    except Exception as e:
        return f"[VQA error] {str(e)}"