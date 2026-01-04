# evaluate.py
import json
import csv
import math

BASELINE_PATH = "baseline.json"
MODEL_PATH = "model_output.json"
OUT_CSV = "evaluation_results.csv"

# ------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

baseline = load_json(BASELINE_PATH)
model = load_json(MODEL_PATH)

# ------------------------------------------------------------
# Cultural signal estimator (continuous, non-hardcoded)
# ------------------------------------------------------------
def cultural_density(text: str) -> float:
    """
    Estimates how strongly cultural context appears in caption.
    Output ∈ [0,1]
    """
    if not text:
        return 0.0

    keywords = [
        "traditional", "festival", "indian",
        "spices", "sweets", "ritual",
        "celebration", "home-style",
        "milk-based", "festive"
    ]

    text = text.lower()
    hits = sum(1 for k in keywords if k in text)

    return min(hits / 4.0, 1.0)


# ------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------
rows = []

for image_id, gt in baseline.items():
    if image_id not in model:
        continue

    pred = model[image_id]

    final_caption = pred["final_caption"]
    predicted_mode = pred["mode"]
    esn_confidence = float(pred.get("esn_confidence", 0.0))

    expected_mode = gt.get("expected_mode", "generic")

    should_have_culture = expected_mode != "generic"
    predicted_culture = predicted_mode != "generic"

    density = cultural_density(final_caption)

    # --------------------------------------------------------
    # 1. Cultural Precision (continuous)
    # --------------------------------------------------------
    if should_have_culture and predicted_culture:
        cultural_precision = esn_confidence * density
    elif not should_have_culture and not predicted_culture:
        cultural_precision = max(0.0, 1.0 - density)
    else:
        cultural_precision = 0.0

    # --------------------------------------------------------
    # 2. Hallucination Score (continuous)
    # --------------------------------------------------------
    if not should_have_culture:
        hallucination_score = density * esn_confidence
    else:
        hallucination_score = max(0.0, 1.0 - density)

    # --------------------------------------------------------
    # 3. Suppression Accuracy
    # --------------------------------------------------------
    suppression_accuracy = 1.0 - hallucination_score

    # --------------------------------------------------------
    # 4. Human Appropriateness (length-based proxy)
    # --------------------------------------------------------
    length = len(final_caption.split())
    human_appropriateness = min(length / 20.0, 1.0)

    # --------------------------------------------------------
    # 5. ESN Usefulness (signed benefit)
    # --------------------------------------------------------
    if should_have_culture and predicted_culture:
        esn_usefulness = esn_confidence * density
    elif not should_have_culture and not predicted_culture:
        esn_usefulness = esn_confidence * (1.0 - density)
    elif not should_have_culture and predicted_culture:
        esn_usefulness = -esn_confidence * density
    else:
        esn_usefulness = 0.0

    # --------------------------------------------------------
    # Composite Metric (paper-weighted)
    # --------------------------------------------------------
    composite_score = (
        0.30 * cultural_precision +
        0.25 * (1.0 - hallucination_score) +
        0.20 * suppression_accuracy +
        0.15 * human_appropriateness +
        0.10 * max(esn_usefulness, 0.0)
    )

    rows.append([
        image_id,
        round(cultural_precision, 3),
        round(hallucination_score, 3),
        round(suppression_accuracy, 3),
        round(human_appropriateness, 3),
        round(esn_usefulness, 3),
        round(composite_score, 3)
    ])

# ------------------------------------------------------------
# Write CSV
# ------------------------------------------------------------
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "cultural_precision",
        "hallucination_score",
        "suppression_accuracy",
        "human_appropriateness",
        "esn_usefulness",
        "composite_score"
    ])
    writer.writerows(rows)

print("✅ Evaluation complete.")
print("Results saved to:", OUT_CSV)
