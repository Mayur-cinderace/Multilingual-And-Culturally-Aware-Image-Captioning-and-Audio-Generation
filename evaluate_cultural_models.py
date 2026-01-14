# evaluate_cultural_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import os
from cultural_context import CulturalContextManager, EnhancedESNCulturalController, EnhancedGRUCulturalController
import random

def generate_training_samples(total=1000):
    samples = []

    food_bases = [
        "Butter chicken", "Paneer tikka masala", "Dal makhani", "Vegetable biryani",
        "Aloo paratha", "Masala dosa", "Rogan josh", "Chole bhature", "Palak paneer",
        "Malai kofta", "Baingan bharta", "Fish curry", "Pork vindaloo", "Chettinad chicken"
    ]
    festival_bases = [
        "Diwali diyas", "Holi colors", "Rangoli design", "Ganesh idol", "Durga Puja pandal",
        "Navratri garba", "Eid biryani", "Karwa Chauth thali", "Onam sadhya", "Pongal kolam"
    ]
    daily_bases = [
        "Morning chai", "Dal rice", "Roti sabzi", "Idli sambar", "Poha breakfast",
        "Curd rice", "Khichdi", "Tiffin lunch", "Evening samosa", "Home-cooked meal"
    ]
    generic_bases = [
        "Mountain lake", "City skyline", "Person reading", "Highway traffic", "Abstract painting",
        "Laptop on desk", "Rainy window", "Bird in flight", "Sunset reflection", "Snowy peak",
        "Empty beach", "Forest path", "Street lamp", "Bicycle parked", "Wooden chair"
    ]

    targets = [
        ("food_traditional", 250, food_bases, ["with naan", "in rich gravy", "served hot", "garnished with coriander", "slow-cooked", "spicy and aromatic"]),
        ("festival_context", 250, festival_bases, ["celebration", "with diyas", "colorful rangoli", "family gathering", "traditional attire", "puja ceremony"]),
        ("daily_life", 250, daily_bases, ["at home", "for breakfast", "family meal", "quick lunch", "evening snack", "homemade"]),
        ("generic", 250, generic_bases, ["at sunset", "in city", "in park", "on highway", "on wall", "at night"])
    ]

    for cls, count, bases, suffixes in targets:
        for _ in range(count):
            base = random.choice(bases)
            suffix = random.choice(suffixes)
            caption = f"{base} {suffix}.".strip()
            samples.append((caption, cls))

    random.shuffle(samples)
    return samples[:total]

# Usage:
training_samples = generate_training_samples(1000)
# ────────────────────────────────────────────────
# Test / Validation data (expand this!)
# ────────────────────────────────────────────────
import random

def generate_large_test_set(size=1000):
    bases = {
        "food_traditional": [
            "Butter chicken", "Paneer tikka masala", "Dal makhani", "Vegetable biryani",
            "Aloo paratha", "Masala dosa", "Rogan josh", "Chole bhature", "Palak paneer",
            "Malai kofta", "Baingan bharta", "Fish curry", "Vindaloo", "Chettinad chicken"
        ],
        "festival_context": [
            "Diwali", "Holi", "Ganesh Chaturthi", "Durga Puja", "Navratri", "Eid",
            "Karwa Chauth", "Onam", "Pongal", "Baisakhi", "Bihu", "Rangoli", "Diya",
            "Prasad", "Modak", "Garba", "Dandiya", "Sheer khurma"
        ],
        "daily_life": [
            "Morning chai", "Filter coffee", "Roti sabzi", "Dal rice", "Poha breakfast",
            "Idli sambar", "Curd rice", "Khichdi", "Tiffin lunch", "Evening snack",
            "Family dinner", "Street vendor", "Home-cooked meal", "Grandmother cooking"
        ],
        "generic": [
            "Mountain landscape", "City skyline", "Person reading", "Car on highway",
            "Abstract painting", "Laptop on desk", "Rain on window", "Bird in tree",
            "Sunset reflection", "Snowy peak", "Open book", "Flower bouquet", "Train journey"
        ]
    }

    test_samples = []
    classes = ["food_traditional", "festival_context", "daily_life", "generic"]
    counts = [250, 250, 250, 250]  # balanced

    for cls, count in zip(classes, counts):
        for _ in range(count):
            base = random.choice(bases[cls])
            if cls == "food_traditional":
                suffix = random.choice(["with naan", "served hot", "garnished with coriander", "in rich gravy", "with rice", "slow-cooked", "spicy and aromatic"])
                caption = f"{base} {suffix}."
            elif cls == "festival_context":
                suffix = random.choice(["celebration", "with diyas", "colorful rangoli", "family gathering", "traditional attire", "puja ceremony", "offered as prasad"])
                caption = f"{base} {suffix}."
            elif cls == "daily_life":
                suffix = random.choice(["at home", "for breakfast", "family meal", "quick lunch", "evening snack", "street style", "homemade"])
                caption = f"{base} {suffix}."
            else:
                suffix = random.choice(["at sunset", "in city", "in park", "on highway", "on wall", "on desk", "at night", "in sky"])
                caption = f"{base} {suffix}."
            
            test_samples.append((caption, cls))

    random.shuffle(test_samples)
    return test_samples

test_samples = generate_large_test_set(1000)

# Output directory for graphs and tables
OUT_DIR = "evaluation_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────
# Cultural signal estimator (from your evaluate.py)
# ────────────────────────────────────────────────
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

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def evaluate_model(model, samples, model_name="Model"):
    """Evaluate ESN or GRU-like object with your custom metrics"""
    true_labels = []
    pred_labels = []
    confidences = []
    injected_flags = []      # for ESN only
    should_have_culture_list = []
    predicted_culture_list = []
    captions_list = []

    mode_to_idx = {mode: i for i, mode in enumerate(model.modes)}
    cultural_modes = ["food_traditional", "festival_context", "daily_life"]  # strings

    for caption, true_mode in samples:
        if true_mode not in model.modes:
            true_mode = "generic"

        # Get prediction
        if isinstance(model, EnhancedESNCulturalController):
            pred_mode, conf = model.predict_mode(caption)
            # Simulate injection (we need this function from app.py / cultural_context.py)
            try:
                _, injection_prefix = get_esn_explanation_and_injection(caption, pred_mode, conf)
                injected = bool(injection_prefix)
            except NameError:
                injected = False  # fallback if function not available
        elif isinstance(model, EnhancedGRUCulturalController):
            pred_mode, conf = model.predict_mode(caption)
            injected = False  # GRU has no injection in current design
        else:
            raise ValueError("Unknown model type")

        true_labels.append(mode_to_idx[true_mode])
        pred_labels.append(mode_to_idx[pred_mode])
        confidences.append(conf)
        injected_flags.append(injected)

        # For custom metrics
        should_have = true_mode in cultural_modes
        predicted_c = pred_mode in cultural_modes
        should_have_culture_list.append(should_have)
        predicted_culture_list.append(predicted_c)
        captions_list.append(caption)

    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    y_conf = np.array(confidences)
    y_injected = np.array(injected_flags)
    should_have_c = np.array(should_have_culture_list)
    predicted_c = np.array(predicted_culture_list)

    metrics = {}

    # 1. Cultural Precision (your logic)
    cultural_precision_values = []
    for i in range(len(samples)):
        if should_have_c[i] and predicted_c[i]:
            density = cultural_density(captions_list[i])
            val = y_conf[i] * density
        elif not should_have_c[i] and not predicted_c[i]:
            density = cultural_density(captions_list[i])
            val = max(0.0, 1.0 - density)
        else:
            val = 0.0
        cultural_precision_values.append(val)
    metrics["cultural_precision"] = np.mean(cultural_precision_values)

    # 2. Hallucination Score (your logic)
    halluc_values = []
    for i in range(len(samples)):
        density = cultural_density(captions_list[i])
        if not should_have_c[i]:
            val = density * y_conf[i]
        else:
            val = max(0.0, 1.0 - density)
        halluc_values.append(val)
    metrics["hallucination_score"] = np.mean(halluc_values)

    # 3. Suppression Accuracy
    metrics["suppression_accuracy"] = 1.0 - metrics["hallucination_score"]

    # 4. Human Appropriateness (your length proxy)
    lengths = [len(c.split()) for c in captions_list]
    ha_values = [min(l / 20.0, 1.0) for l in lengths]
    metrics["human_appropriateness"] = np.mean(ha_values)

    # 5. ESN Usefulness (your signed version)
    usefulness_values = []
    for i in range(len(samples)):
        density = cultural_density(captions_list[i])
        if should_have_c[i] and predicted_c[i]:
            val = y_conf[i] * density
        elif not should_have_c[i] and not predicted_c[i]:
            val = y_conf[i] * (1.0 - density)
        elif not should_have_c[i] and predicted_c[i]:
            val = - y_conf[i] * density
        else:
            val = 0.0
        usefulness_values.append(val)
    metrics["esn_usefulness"] = np.mean(usefulness_values)

    # 6. Composite Score (your exact weights)
    composite = (
        0.30 * metrics["cultural_precision"] +
        0.25 * (1.0 - metrics["hallucination_score"]) +
        0.20 * metrics["suppression_accuracy"] +
        0.15 * metrics["human_appropriateness"] +
        0.10 * max(metrics["esn_usefulness"], 0.0)
    )
    metrics["composite_score"] = composite

    return {
        "true": y_true,
        "pred": y_pred,
        "conf": y_conf,
        "injected": y_injected,
        "metrics": metrics,
        "captions": captions_list,
        "modes": model.modes
    }


# ────────────────────────────────────────────────
# The rest remains almost identical (tables + plots)
# ────────────────────────────────────────────────
def generate_evaluation_table(results, model_name):
    metrics = results["metrics"]
    df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": [f"{v:.4f}" for v in metrics.values()]
    })
    print(f"\n=== {model_name} Custom Metrics Table ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, f"{model_name.lower()}_custom_metrics.csv"), index=False)
    return df


def plot_confusion_matrix(results, model_name):
    cm = confusion_matrix(results["true"], results["pred"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(results["modes"])))
    ax.set_yticks(range(len(results["modes"])))
    ax.set_xticklabels(results["modes"], rotation=45)
    ax.set_yticklabels(results["modes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{model_name.lower()}_confusion_matrix.png"))
    plt.close()
    print(f"Saved confusion matrix for {model_name}")


def plot_metric_bar(results_esn, results_gru):
    metrics_esn = results_esn["metrics"]
    metrics_gru = results_gru["metrics"]
    metric_names = list(metrics_esn.keys())
    esn_scores = [metrics_esn[m] for m in metric_names]
    gru_scores = [metrics_gru[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, esn_scores, width, label="ESN")
    ax.bar(x + width/2, gru_scores, width, label="GRU")
    ax.set_ylabel("Score")
    ax.set_title("Custom Metric Comparison: ESN vs GRU")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "custom_metric_comparison.png"))
    plt.close()
    print(f"Saved custom metric comparison bar plot")


def plot_confidence_distribution(results, model_name):
    correct = (results["true"] == results["pred"])
    conf_correct = results["conf"][correct]
    conf_incorrect = results["conf"][~correct]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(conf_correct, bins=10, alpha=0.7, label="Correct", color="green")
    ax.hist(conf_incorrect, bins=10, alpha=0.7, label="Incorrect", color="red")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{model_name.lower()}_confidence_dist.png"))
    plt.close()
    print(f"Saved confidence distribution for {model_name}")


def print_evaluation_report(results, model_name):
    y_true = results["true"]
    y_pred = results["pred"]
    confs = results["conf"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {model_name} Standard Evaluation ===")
    print(f"Accuracy: {acc:.4f}  ({sum(y_true == y_pred)} / {len(y_true)})")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=results["modes"],
        zero_division=0
    ))

    print("\nConfusion Matrix (numeric):")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nConfidence stats:")
    print(f"  Mean:   {np.mean(confs):.4f}")
    print(f"  Median: {np.median(confs):.4f}")
    print(f"  Range:  {np.min(confs):.4f} – {np.max(confs):.4f}")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing models and training on training_samples...")
    cultural_manager = CulturalContextManager()
    cultural_manager.train(training_samples)  # re-trains each time

    print(f"\nEvaluating on {len(test_samples)} test examples...\n")

    # ESN
    esn_results = evaluate_model(cultural_manager.esn, test_samples, "ESN")
    print_evaluation_report(esn_results, "ESN")
    generate_evaluation_table(esn_results, "ESN")
    plot_confusion_matrix(esn_results, "ESN")
    plot_confidence_distribution(esn_results, "ESN")

    # GRU
    gru_results = evaluate_model(cultural_manager.gru, test_samples, "GRU")
    print_evaluation_report(gru_results, "GRU")
    generate_evaluation_table(gru_results, "GRU")
    plot_confusion_matrix(gru_results, "GRU")
    plot_confidence_distribution(gru_results, "GRU")

    # Head-to-head
    print("\n=== ESN vs GRU Head-to-Head ===")
    agreements = sum(esn_results["pred"] == gru_results["pred"])
    print(f"Prediction agreement: {agreements}/{len(test_samples)} = {agreements/len(test_samples):.1%}")

    both_correct = sum((esn_results["pred"] == esn_results["true"]) & 
                       (gru_results["pred"] == gru_results["true"]))
    esn_only_correct = sum((esn_results["pred"] == esn_results["true"]) & 
                           (gru_results["pred"] != gru_results["true"]))
    gru_only_correct = sum((esn_results["pred"] != esn_results["true"]) & 
                           (gru_results["pred"] == gru_results["true"]))

    print(f"Both correct      : {both_correct}")
    print(f"Only ESN correct  : {esn_only_correct}")
    print(f"Only GRU correct  : {gru_only_correct}")

    # Comparison plots
    plot_metric_bar(esn_results, gru_results)

    print(f"\nAll outputs saved to: {OUT_DIR}/")