import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None

def load_refiner():
    global _tokenizer, _model
    if _model is None:
        print("[LLM] Loading Flan-T5 (lazy)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).eval()
    return _tokenizer, _model


def unload_refiner():
    global _tokenizer, _model
    if _model is not None:
        print("[LLM] Unloading Flan-T5")
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        torch.cuda.empty_cache()
