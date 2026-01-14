# evaluate_caption_metrics.py
"""
Standalone script to compute classical captioning metrics (BLEU, METEOR, ROUGE, CIDEr, BERTScore)
between generated captions and reference (ground truth) captions.

No cultural mode / ESN / GRU logic here — pure caption quality evaluation.
"""

import json
import os
from collections import defaultdict
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import SmoothingFunction
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except ImportError:
    print("Please install nltk:  pip install nltk")
    exit(1)

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Please install rouge-score:  pip install rouge-score")
    exit(1)

# Optional — better semantic metric (needs pip install bert-score)
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("BERTScore not installed → skipping (pip install bert-score to enable)")

# Optional — CIDEr (needs pycocoevalcap from COCO eval repo)
HAS_CIDER = False
try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except ImportError:
    print("pycocoevalcap not installed → CIDEr skipped")
    print("To enable CIDEr: git clone https://github.com/salaniz/pycocoevalcap && cd pycocoevalcap && pip install -e .")

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
RESULTS_JSON = "caption_results.json"          # your generated outputs
REFERENCES_JSON = "references.json"            # ground truth captions

# Format expected in both JSON files:
# {
#   "image_001": ["A bowl of spicy curry with herbs."],
#   "image_002": ["People celebrating Diwali with lights and sweets."],
#   ...
# }
# References can have multiple captions per image (list), generated usually has one.

OUT_CSV = "caption_metrics_results.csv"
OUT_TXT = "caption_metrics_summary.txt"

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────
def load_captions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

gen_captions = load_captions(RESULTS_JSON)
ref_captions = load_captions(REFERENCES_JSON)

# Align keys (only evaluate images present in both)
common_ids = set(gen_captions.keys()) & set(ref_captions.keys())
print(f"Evaluating {len(common_ids)} images present in both files.")

# Prepare lists for scorers
gts = defaultdict(list)   # reference
res = {}                  # generated

for img_id in common_ids:
    gen = gen_captions[img_id]
    refs = ref_captions[img_id]

    # Make sure generated is list of one string
    if isinstance(gen, str):
        gen = [gen]
    res[img_id] = gen

    # References can be list or single string
    if isinstance(refs, str):
        refs = [refs]
    gts[img_id] = refs

# ────────────────────────────────────────────────
# Compute metrics
# ────────────────────────────────────────────────
bleu_scores = {1: [], 2: [], 3: [], 4: []}
meteor_scores = []
rouge_scores = []
bert_scores = [] if HAS_BERTSCORE else None
cider_score = None

smooth = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

for img_id in common_ids:
    gen_list = res[img_id]           # usually length 1
    ref_list = gts[img_id]           # can be multiple

    gen = gen_list[0]  # take first (only) generated caption

    # BLEU
    ref_tokens = [r.split() for r in ref_list]
    gen_tokens = gen.split()
    for n in [1,2,3,4]:
        bleu = sentence_bleu(ref_tokens, gen_tokens, weights=(1/n,)*(n) + (0,)*(4-n), smoothing_function=smooth)
        bleu_scores[n].append(bleu)

    # METEOR
    meteor = meteor_score(ref_tokens, gen_tokens)
    meteor_scores.append(meteor)

    # ROUGE-L
    rouge = scorer.score(' '.join(ref_list[0].split()), gen)['rougeL'].fmeasure
    rouge_scores.append(rouge)

    # BERTScore (optional)
    if HAS_BERTSCORE:
        P, R, F1 = bert_score([gen], [' '.join(ref_list[0].split())], lang="en", verbose=False)
        bert_scores.append(F1.mean().item())

# CIDEr (corpus level)
if HAS_CIDER:
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    cider_score = score

# ────────────────────────────────────────────────
# Aggregate & print
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("Captioning Metrics Summary")
print("="*60)

avg_bleu = {n: np.mean(bleu_scores[n]) for n in [1,2,3,4]}
print("BLEU:")
for n in [1,2,3,4]:
    print(f"  BLEU-{n}: {avg_bleu[n]:.4f}")

print(f"METEOR:     {np.mean(meteor_scores):.4f}")
print(f"ROUGE-L:    {np.mean(rouge_scores):.4f}")

if HAS_BERTSCORE:
    print(f"BERTScore:  {np.mean(bert_scores):.4f}")

if HAS_CIDER:
    print(f"CIDEr:      {cider_score:.4f}")

print("="*60)

# Save per-image results to CSV
rows = []
for img_id in common_ids:
    gen = res[img_id][0]
    ref = ' || '.join(gts[img_id])  # join multiple refs
    bleu1 = sentence_bleu([r.split() for r in gts[img_id]], gen.split(), weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = sentence_bleu([r.split() for r in gts[img_id]], gen.split(), weights=(0,0,0,1), smoothing_function=smooth)
    meteor = meteor_score([r.split() for r in gts[img_id]], gen.split())
    rouge = scorer.score(' '.join(gts[img_id][0].split()), gen)['rougeL'].fmeasure

    rows.append([img_id, gen, ref, f"{bleu1:.4f}", f"{bleu4:.4f}", f"{meteor:.4f}", f"{rouge:.4f}"])

import csv
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "generated", "reference", "BLEU-1", "BLEU-4", "METEOR", "ROUGE-L"])
    writer.writerows(rows)

print(f"Per-image results saved to: {OUT_CSV}")
print(f"Summary saved to: {OUT_TXT} (you can redirect console output if needed)")

# Optional: save summary to text file
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write("Captioning Metrics Summary\n")
    f.write("="*50 + "\n")
    for n in [1,2,3,4]:
        f.write(f"BLEU-{n}: {avg_bleu[n]:.4f}\n")
    f.write(f"METEOR:     {np.mean(meteor_scores):.4f}\n")
    f.write(f"ROUGE-L:    {np.mean(rouge_scores):.4f}\n")
    if HAS_BERTSCORE:
        f.write(f"BERTScore:  {np.mean(bert_scores):.4f}\n")
    if HAS_CIDER:
        f.write(f"CIDEr:      {cider_score:.4f}\n")