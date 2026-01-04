"""
categorize_and_tag.py
Auto-categorize images into cultural categories using CLIP zero-shot.
Outputs:
 - curate_categorized/<category>/*.jpg  (copied images)
 - metadata.csv  (image, original_path, assigned_categories, top_tags (json), confidences)
 - metadata.json (detailed per-image info)
"""
import os

# MUST COME FIRST BEFORE ANY TRANSFORMERS IMPORT
os.environ["HF_HOME"] = r"D:\ANNDL\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\ANNDL\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\ANNDL\hf_cache"

import sys, json, shutil, math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
import open_clip
import numpy as np

# ------------- CONFIG -------------
ROOT = Path("curate")                # your current dataset folder
OUT = Path("curate_categorized")     # where categorized images go
OUT.mkdir(exist_ok=True)
MIN_CONF = 0.20     # min CLIP score to accept a tag (0..1, tune)
TOPK_TAGS = 8       # store top K tags for metadata
COPY_INSTEAD_OF_MOVE = True  # set False to move files instead of copying
USE_GPU = torch.cuda.is_available()

# Cultural categories and textual prompts (expandable)
CATEGORIES = {
    "temple": [
        "a hindu temple exterior",
        "a hindu temple interior",
        "a religious shrine in India",
        "a temple gopuram"
    ],
    "market": [
        "a busy indian street market",
        "a bazaar with stalls and vegetables",
        "a rupee lined street market"
    ],
    "railway_station": [
        "an indian railway platform",
        "a busy train station in India"
    ],
    "kitchen_food": [
        "an Indian kitchen with cooking utensils",
        "a plate of Indian food",
        "a thali with multiple Indian dishes",
        "street food stall serving Indian food"
    ],
    "people_attire": [
        "a person wearing a saree",
        "a person wearing a kurta",
        "a person wearing a turban",
        "traditional Indian clothing"
    ],
    "festival": [
        "people celebrating holi with colors",
        "a diwali scene with diyas and lamps",
        "a wedding ceremony in India"
    ],
    "village": [
        "a rural indian village street",
        "mud houses in indian village",
        "a farmer working in field"
    ],
    "street_traffic": [
        "a busy indian street with rickshaws and motorcycles",
        "auto rickshaw on indian street"
    ],
    "objects_cultural": [
        "a rangoli",
        "a tiffin box",
        "a banana leaf with food",
        "a temple bell",
        "a diya (oil lamp)"
    ],
    "food_specific": [  # separate more granular food tags
        "idli",
        "dosa",
        "biryani",
        "samosa",
        "jalebi",
        "gulab jamun",
        "thali"
    ]
}

# Flatten a list of textual candidate labels (tags) with grouping info
CANDIDATES = []
CAT_MAP = {}   # map tag index -> category_name
for cat, prompts in CATEGORIES.items():
    for p in prompts:
        CAT_MAP[len(CANDIDATES)] = cat
        CANDIDATES.append(p)

print(f"Total textual candidates: {len(CANDIDATES)}")

# ------------- Load CLIP model -------------
device = "cuda" if USE_GPU else "cpu"
print("Using device:", device)
model_name = "ViT-L-14"   # large model if you have VRAM; fallback to ViT-B/32 if not
pretrained = "laion2b_s32b_b79k"  # or "openai" depending on availability

# Try to load a model that fits memory; if fails, fallback
try:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
except Exception as e:
    print("Large model load failed, falling back to smaller model. Error:", e)
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

model.to(device).eval()

# encode candidate text
with torch.no_grad():
    text_tokens = tokenizer(CANDIDATES)
    text_embeddings = model.encode_text(text_tokens.to(device))
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# ------------- Helpers -------------
def list_images(root_dir):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in Path(root_dir).rglob("*"):
        if p.suffix.lower() in img_exts:
            yield p

def score_image(img_path):
    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print("Skipping (cannot open):", img_path, "err:", e)
        return None
    with torch.no_grad():
        img_emb = model.encode_image(img)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        # cosine similarity
        sims = (100.0 * img_emb @ text_embeddings.T).squeeze(0)  # scaled logits
        probs = sims.softmax(dim=0).cpu().numpy()   # probability-like over candidates
        # get topk
        topk_idx = np.argsort(-probs)[:TOPK_TAGS]
        results = [(int(i), float(probs[i])) for i in topk_idx]
        return results

# ------------- Main loop -------------
rows = []
total = 0
for img_path in tqdm(list(list_images(ROOT)), desc="Scanning images"):
    total += 1
    res = score_image(img_path)
    if res is None:
        continue

    # map candidate indexes back to category names and aggregate scores per category
    cat_scores = {}
    tag_list = []
    for idx, score in res:
        tag_text = CANDIDATES[idx]
        cat = CAT_MAP[idx]
        cat_scores.setdefault(cat, 0.0)
        cat_scores[cat] += score  # sum evidence for the category
        tag_list.append({"text": tag_text, "score": score, "cat": cat})

    # decide categories to assign based on aggregated score
    assigned = []
    for cat, s in cat_scores.items():
        # normalize? we use threshold
        if s >= MIN_CONF:
            assigned.append((cat, s))
    # if nothing passes threshold, still pick best category (fallback)
    if not assigned:
        best_cat = max(cat_scores.items(), key=lambda x: x[1])[0]
        assigned = [(best_cat, cat_scores[best_cat])]

    assigned_names = [c for c,_ in assigned]

    # copy image to each assigned category folder (multilabel)
    for cat_name in assigned_names:
        dst_dir = OUT / cat_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / img_path.name
        try:
            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(img_path, dst_file)
            else:
                shutil.move(img_path, dst_file)
        except FileExistsError:
            # if file exists, optionally rename (append index)
            base, ext = os.path.splitext(dst_file.name)
            i = 1
            while (dst_dir / f"{base}_{i}{ext}").exists():
                i += 1
            newname = dst_dir / f"{base}_{i}{ext}"
            shutil.copy2(img_path, newname)

    # save metadata row
    row = {
        "image": str(img_path),
        "filename": img_path.name,
        "assigned_categories": assigned_names,
        "category_scores": {k: v for k,v in cat_scores.items()},
        "top_tags": tag_list
    }
    rows.append(row)

# ------------- Save metadata -------------
df = pd.DataFrame(rows)
df.to_csv("metadata.csv", index=False)
with open("metadata.json", "w", encoding="utf8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Processed {total} images. Output folders in: {OUT}")
