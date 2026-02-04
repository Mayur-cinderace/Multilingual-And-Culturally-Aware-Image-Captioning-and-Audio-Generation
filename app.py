# app.py – Cultural AI Explorer with SQLite DB + Ollama (local)
import gradio as gr
from gtts import gTTS
from pathlib import Path
import speech_recognition as sr
import os
import sqlite3
import json
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
import datetime
from datetime import datetime, timezone
import numpy as np
from esn_gru_visualization import (
    animate_esn_reservoir,
    compare_esn_gru_3d,
    counterfactual_drift_animation
)

load_dotenv()

# ────────────────────────────────────────────────
#   LOCAL OLLAMA CLIENT
# ────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"  # change to your preferred model: qwen2.5, llama3.2, etc.

def ollama_generate(prompt, system="", temperature=0.7, max_tokens=300):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            output = ""
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "message" in data:
                    output += data["message"]["content"]
                if data.get("done"):
                    break
            return output.strip()
    except Exception as e:
        return f"[Ollama error] {str(e)}"


# ------------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------------
DB_PATH = "cultural_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS cultural_explorer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT,
            confidence REAL,
            english_caption TEXT,
            local_caption TEXT,
            hashtags TEXT,

            -- NEW: schema-less signal storage
            signal_vector TEXT,          -- JSON
            signal_summary TEXT,         -- short human-readable summary

            added_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------
# OTHER SETUP
# ------------------------------------------------------------
OUT_AUDIO = Path("captions_output_cpu/audio")
OUT_AUDIO.mkdir(parents=True, exist_ok=True)

TTS_LANG = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa",
    "Urdu": "ur", "Odia": "or", "Assamese": "as", "Nepali": "ne", "Konkani": "gom", "Sanskrit": "sa"
}

CLUSTER_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f",
    "#f4a261", "#8d99ae", "#6a4c93"
]

from cpu_cultural_enrich import generate_base_caption, translate_to_language
from cultural_context import CulturalContextManager
from hashtag_generator import generate_hashtags
from translation_refiner import naturalize_translation

training_samples = [
    # food_traditional  ── 10 examples
    ("A bowl of spicy curry garnished with fresh herbs.", "food_traditional"),
    ("Aromatic masala dish served in a traditional bowl.", "food_traditional"),
    ("Gravy with spices and vegetables.", "food_traditional"),
    ("Spiced rice garnished with nuts.", "food_traditional"),
    ("Traditional meal with rich flavors.", "food_traditional"),
    ("Creamy paneer butter masala with garlic naan.", "food_traditional"),
    ("Dal tadka served with jeera rice.", "food_traditional"),
    ("Vegetable korma in coconut gravy.", "food_traditional"),
    ("Chicken tikka masala with butter naan.", "food_traditional"),
    ("Aloo gobi dry sabzi with roti.", "food_traditional"),

    # festival_context  ── 8 examples
    ("People celebrating with lights and sweets during festival.", "festival_context"),
    ("Ritual offerings in a celebration.", "festival_context"),
    ("Festival gathering with colorful decorations.", "festival_context"),
    ("Traditional dance during celebratory event.", "festival_context"),
    ("Diwali diyas and rangoli at home entrance.", "festival_context"),
    ("Holi colors thrown in joyful gathering.", "festival_context"),
    ("Ganesh idol decorated with flowers and modak.", "festival_context"),
    ("Navratri garba dance in traditional clothes.", "festival_context"),

    # daily_life  ── 4 examples (smaller because it's closer to generic)
    ("Family having a simple home meal.", "daily_life"),
    ("Daily routine with breakfast in bowl.", "daily_life"),
    ("Home-cooked dish served to family.", "daily_life"),
    ("Everyday gathering around the table.", "daily_life"),

    # generic  ── 18 examples  (≈45%)
    ("A landscape with mountains and sky.", "generic"),
    ("Abstract art piece on wall.", "generic"),
    ("City street with cars.", "generic"),
    ("Book on a table.", "generic"),
    ("Random object in room.", "generic"),
    ("Sunset over calm ocean.", "generic"),
    ("Modern office building exterior.", "generic"),
    ("Person walking in park.", "generic"),
    ("Highway traffic at dusk.", "generic"),
    ("Clouds in blue sky.", "generic"),
    ("Cat sleeping on windowsill.", "generic"),
    ("Laptop and notebook on desk.", "generic"),
    ("Train passing through countryside.", "generic"),
    ("Flower pot on balcony.", "generic"),
    ("Snow-covered trees in winter.", "generic"),
    ("Empty beach at sunrise.", "generic"),
    ("Street lamp at night.", "generic"),
    ("Bicycle parked near wall.", "generic"),
]

def augment_samples(samples):
    augmented = []

    templates = [
        "An image showing {}",
        "A scene with {}",
        "A close-up of {}",
        "A casual view of {}",
        "A detailed shot of {}",
        "A setting that includes {}"
    ]

    noise_suffixes = [
        "",
        " on a table",
        " indoors",
        " under natural light",
        " during the day",
        " in a home setting",
        " with other items nearby"
    ]

    for caption, mode in samples:
        augmented.append((caption, mode))

        # light paraphrases
        for t in templates:
            augmented.append((t.format(caption.lower()), mode))

        # noisy / partial versions
        words = caption.split()
        if len(words) > 5:
            truncated = " ".join(words[:len(words)//2])
            augmented.append((truncated, mode))

        # ambiguity injection
        augmented.append((caption + noise_suffixes[np.random.randint(len(noise_suffixes))], mode))

    return augmented


cultural_manager = CulturalContextManager()
augmented_samples = augment_samples(training_samples)
cultural_manager.train(augmented_samples)


CULTURAL_CONFIDENCE_THRESHOLD = 0.55

CULTURAL_TEMPLATES = {
    "food_traditional": "This reflects traditional cooking styles, often featuring rich spices and shared meals.",
    "festival_context": "Such scenes are common in cultural festivals and communal celebrations.",
    "daily_life": "This is typically part of everyday routines and family interactions.",
    "generic": ""
}

ESN_FEATURE_NAMES = [
    "Caption Length", "Spice Level", "Ritual Signals", "Daily Life",
    "Heritage", "Color", "Texture", "Serving Style",
    "Has Thali", "Festival Mention", "Sweet", "Rice",
    "Vegetarian", "Meat", "Dessert", "Beverage",
    "Offering", "Decoration", "People", "Utensils"
]

COUNTERFACTUAL_FRAMES = {
    "Post-pandemic urban": {
        "people_present": 0.6,
        "daily_life": 1.2
    },
    "Rural 1980s": {
        "heritage_signal": 1.3,
        "object_focus": 0.7
    },
    "Minimalist modern": {
        "people_present": 0.5,
        "object_focus": 1.4,
        "ritual_signal": 0.6
    }
}


# ── Temporal decay configuration ─────────────────────────────
TEMPORAL_DECAY_LAMBDA = 0.12   # higher = faster forgetting


def extract_signal_vector(caption: str) -> dict:
    feats = cultural_manager.esn.extract_features(caption)

    return {
        "people_present": float(feats[18]),
        "ritual_signal": float(feats[2]),
        "daily_life": float(feats[3]),
        "food_signal": float(feats[11]),
        "heritage_signal": float(feats[4]),
        "object_focus": float(feats[7]),
        "festival_signal": float(feats[9]),
        "has_thali": float(feats[8]),  
    }

def summarize_signals(signal_vec: dict) -> str:
    active = [
        k.replace("_", " ")
        for k, v in signal_vec.items()
        if v >= 0.65
    ]
    return ", ".join(active) if active else "No dominant cultural signals"

def counterfactual_from_vector(signal_vector: dict, frame: str) -> dict:
    """
    Apply counterfactual cultural frame directly on stored signal vector
    (no re-captioning, no re-vision noise).
    """

    if not signal_vector:
        return {"error": "No cultural signal available yet."}

    modifiers = COUNTERFACTUAL_FRAMES.get(frame, {})
    adjusted = {}

    for k, v in signal_vector.items():
        adjusted[k] = round(
            min(1.0, v * modifiers.get(k, 1.0)), 3
        )

    dominant = sorted(adjusted.items(), key=lambda x: -x[1])[:3]

    return {
        "frame": frame,
        "original_signals": signal_vector,
        "adjusted_signals": adjusted,
        "dominant_dimensions": dominant,
        "interpretation": (
            f"Under the '{frame}' frame, cultural emphasis shifts toward "
            f"{', '.join(k for k, _ in dominant)}."
        )
    }


def build_cultural_signal_plot(caption: str):
    feats = cultural_manager.esn.extract_features(caption)

    indices = [1, 2, 3, 7, 9, 11, 18]
    labels = [ESN_FEATURE_NAMES[i] for i in indices]
    values = [float(feats[i]) for i in indices]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, values, color="#f4a261")
    ax.set_xlabel("Activation Strength")
    ax.set_title("Cultural Signal Strength (ESN)")
    plt.tight_layout()
    return fig

def build_cultural_tag_cloud(caption: str):
    feats = cultural_manager.esn.extract_features(caption)

    tags = []
    for name, idx in {
        "People present": 18,
        "Ritual cues": 2,
        "Daily activity": 3,
        "Heritage cues": 4,
        "Object focus": 7,
        "Festival context": 9,
    }.items():
        if feats[idx] >= 0.65:
            tags.append(name)

    return " • ".join(tags) if tags else "No dominant cultural tags detected."


def build_cultural_index(signal_vector: dict) -> dict:
    """
    Produces a normalized archival index usable for museums/newsrooms.
    Values are clamped to [0,1].
    """
    index = {}
    for k, v in signal_vector.items():
        index[k] = round(min(1.0, max(0.0, float(v))), 3)
    return index

def extract_mode_from_prediction(pred):
    if not pred or not isinstance(pred, dict):
        return "generic"
    return pred.get("combined_mode", "generic")

def cultural_entropy(features: np.ndarray, eps=1e-9) -> float:
    """
    Measures dispersion of cultural signals.
    High entropy = no dominant cultural interpretation.
    """
    x = np.abs(features.astype(float))
    x = x / (np.sum(x) + eps)
    entropy = -np.sum(x * np.log(x + eps))
    return float(entropy)

def detect_cultural_surprise(caption: str, prediction: dict) -> dict:
    """
    Detects violation of learned cultural expectations.
    """

    feats = cultural_manager.esn.extract_features(caption)
    entropy = cultural_entropy(feats)

    esn = prediction["esn"]
    gru = prediction["gru"]

    disagreement = esn["mode"] != gru["mode"]
    confidence_gap = abs(esn["confidence"] - gru["confidence"])
    overall_conf = max(esn["confidence"], gru["confidence"])

    surprise = (
        disagreement and
        entropy > 2.2 and
        0.35 <= overall_conf <= 0.7 and
        confidence_gap > 0.2
    )


    explanation = ""
    if surprise:
        explanation = (
            "The image activates multiple conflicting cultural signals. "
            "Immediate visual cues and learned cultural patterns disagree, "
            "indicating a break from dominant expectations."
        )

    if not surprise:
        explanation = "No cultural expectation violation detected."

    return {
        "is_surprise": surprise,
        "entropy": round(entropy, 3),
        "esn_mode": esn["mode"],
        "gru_mode": gru["mode"],
        "confidence": round(overall_conf, 3),
        "explanation": explanation
    }



# ------------------------------------------------------------
# DB Helper Functions
# ------------------------------------------------------------
def get_cultural_facts(mode: str) -> str:
    if mode == "generic":
        return "_No relevant cultural memory for this scene._"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT signal_summary, english_caption
        FROM cultural_explorer
        WHERE mode = ?
        ORDER BY added_at DESC
        LIMIT 8
    """, (mode,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return "No cultural memory yet."

    return "\n".join(
        f"- **{s}**"
        for s, _ in rows if s
    )


def add_cultural_entry(mode, name, description, region, festival, story, recipe_summary):
    try:
        signal_summary = ", ".join(
            x for x in [name, description, region, festival] if x
        )[:120]

        payload = {
            "name": name,
            "description": description,
            "region": region,
            "festival": festival,
            "story": story,
            "recipe_summary": recipe_summary
        }

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO cultural_explorer (
                mode, confidence, english_caption,
                signal_vector, signal_summary
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            mode,
            0.0,
            name or description or "Manual cultural entry",
            json.dumps(payload),
            signal_summary
        ))
        conn.commit()
        conn.close()
        return "Entry added successfully!"
    except Exception as e:
        return f"Error adding entry: {str(e)}"
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_esn_explanation_and_injection(caption: str, mode: str, conf: float, top_k=4) -> tuple[str, str]:
    """
    Returns:
    - short explanation line (e.g. "ESN detected strong festival + thali signals")
    - modified sentence piece to inject (or "" if weak)
    """
    if conf < 0.52:
        return "", ""

    feats = cultural_manager.esn.extract_features(caption)  # assuming you can access it
    feature_names = [
        "caption_length", "spice_level", "ritual_score", "daily_score", "heritage",
        "color_vivid", "texture_words", "serving_style", "has_thali", "has_festival",
        "has_sweet", "has_rice", "veg_focus", "meat_focus", "has_dessert", "has_beverage",
        "has_offering", "has_decoration", "has_people", "has_utensil",
        "adj_ratio", "uniq_ratio", "sentiment_pos"
    ] + ["bias"] * 5  # last few are constants

    # Get top indices (ignore last 5 bias terms)
    top_indices = np.argsort(feats[:-5])[::-1][:top_k]
    top_pairs = [(feature_names[i], round(float(feats[i]), 2)) for i in top_indices if feats[i] > 0.35]

    if not top_pairs:
        return "", ""

    # Build explanation
    strongest = top_pairs[0][0]
    explanation = f"ESN activated mainly by: " + ", ".join(f"{name} ({val})" for name, val in top_pairs[:3])

    # Decide what stylistic injection to make
    injection = ""
    if "has_festival" in strongest or any("festival" in n for n, v in top_pairs if v > 1.2):
        injection = "festive "
    elif "has_thali" in strongest or feats[feature_names.index("has_thali")] > 1.4:
        injection = "traditional thali of "
    elif "spice_level" in strongest and top_pairs[0][1] > 1.1:
        injection = "richly spiced "
    elif "has_offering" in strongest or "ritual" in strongest:
        injection = "ceremonial "
    elif "heritage" in strongest and top_pairs[0][1] > 0.9:
        injection = "time-honored "
    elif "daily_score" in strongest and top_pairs[0][1] > 0.9:
        injection = "comforting homemade "

    return explanation, injection
    
import re
from collections import Counter

STOPWORDS = {
    "the","a","an","with","and","or","of","in","on","at","to","for","from",
    "this","that","is","are","was","were","shows","showing","image","picture",
    "reflects","featuring","often","style","styles","scene","view"
}

def build_pexels_query(caption: str, max_terms: int = 5) -> str:
    """
    Convert a full caption into a short, search-safe keyword query.
    Generic, domain-independent, no hardcoding.
    """

    if not caption:
        return "photo"

    # lowercase + remove punctuation
    text = caption.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = text.split()

    # remove stopwords + very short tokens
    tokens = [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 3
    ]

    if not tokens:
        return "photo"

    # frequency-based salience
    freq = Counter(tokens)

    # prefer mid-caption nouns (avoid boilerplate starts)
    sorted_tokens = sorted(
        freq.items(),
        key=lambda x: (-x[1], tokens.index(x[0]))
    )

    keywords = [w for w, _ in sorted_tokens[:max_terms]]

    return " ".join(keywords)

def detect_cultural_silence(caption: str) -> dict:
    feats = cultural_manager.esn.extract_features(caption)

    silence_axes = {
        "people_absent": feats[18] < 0.3,
        "ritual_absent": feats[2] < 0.25,
        "communal_absent": feats[18] < 0.3,
        "heritage_absent": feats[4] < 0.3,
    }

    active_silences = [k for k, v in silence_axes.items() if v]

    if len(active_silences) < 2:
        return {"is_silence": False}

    interpretation = (
        "The image is marked by the absence of social and ritual cues, "
        "suggesting isolation, neutrality, transition, or private space."
    )

    if len(active_silences) < 2:
        return {
            "is_silence": False,
            "missing_signals": [],
            "interpretation": "No significant cultural absence detected."
        }

    return {
        "is_silence": True,
        "missing_signals": active_silences,
        "interpretation": interpretation
    }


def search_similar_images(caption: str, num_results=8):
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_API_KEY:
        return ["https://via.placeholder.com/300x200?text=Add+PEXELS_API_KEY+in+.env"]

    query = build_pexels_query(caption)

    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": num_results
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        photos = data.get("photos", [])
        urls = [p["src"]["medium"] for p in photos if "src" in p]

        return urls or ["https://via.placeholder.com/300x200?text=No+Images+Found"]

    except Exception as e:
        print(f"[Pexels] {e}")
        return ["https://via.placeholder.com/300x200?text=Error+Loading+Images"]


# ------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------
def run_caption(image, language, use_esn, compare_models=False):
    """
    Returns (IN THIS ORDER):
    1. final_caption_en
    2. local_caption
    3. hashtags_str
    4. audio_file
    5. image_path
    6. context_state
    7. comparison_text
    8. reasoning_trace
    9. prediction_results
    10. cultural_signal_plot
    11. cultural_tag_cloud
    12. cultural_index
    13. signal_vector
    14. disagreement_report
    15. surprise_report
    16. silence_report
    """

    # ── Safety: no image ──────────────────────────────────────
    if image is None:
        return (
            "",                     # final_caption_en
            "",                     # local_caption
            "",                     # hashtags
            None,                   # audio
            None,                   # image_path
            None,                   # context_state
            "",                     # comparison
            "No image provided.",   # reasoning
            None,                   # prediction
            None,                   # plot
            "",                     # tag cloud
            {},                     # cultural index
            {},                     # signal vector
            {},                     # disagreement
            {},                     # surprise
            {}                      # silence
        )


    # ── Save image ────────────────────────────────────────────
    image_path = "temp.jpg"
    image.save(image_path)

    # ── Base caption (vision-grounded) ─────────────────────────
    base_caption = generate_base_caption(image_path).strip()

    # ── Cultural classification ────────────────────────────────
    results = cultural_manager.predict(base_caption)
    # ── ESN / GRU COGNITION VISUALIZATIONS ─────────────────────
    try:
        esn_evolution_html = animate_esn_reservoir(
            base_caption,
            cultural_manager.esn,
            output_path="static/esn_evolution.html"
        )

        esn_vs_gru_html = compare_esn_gru_3d(
            base_caption,
            cultural_manager.esn,
            cultural_manager.gru,
            output_path="static/esn_vs_gru.html"
        )

        counterfactual_html = counterfactual_drift_animation(
            base_caption,
            cultural_manager.esn,
            COUNTERFACTUAL_FRAMES,
            output_path="static/counterfactual_drift.html"
        )

    except Exception as e:
        print("[Visualization error]", e)

    mode = results["combined_mode"]
    esn_conf = results["esn"]["confidence"]
    gru_conf = results["gru"]["confidence"]
    overall_conf = max(esn_conf, gru_conf)
    surprise_report = detect_cultural_surprise(base_caption, results)
    silence_report = detect_cultural_silence(base_caption)

    # ── ESN explanation + injection ────────────────────────────
    esn_explain = ""
    injection_prefix = ""

    if use_esn:
        esn_explain, injection_prefix = get_esn_explanation_and_injection(
            base_caption, mode, esn_conf
        )

    cultural_suffix = CULTURAL_TEMPLATES.get(mode, "")

    # ── Final English caption ──────────────────────────────────
    if injection_prefix and overall_conf >= CULTURAL_CONFIDENCE_THRESHOLD:
        final_caption_en = (
            f"{injection_prefix}{base_caption.rstrip('.')}. {cultural_suffix}"
        ).strip()
        reasoning_trace = f"Cultural enhancement applied by ESN.\n{esn_explain}"
    else:
        final_caption_en = (
            base_caption.rstrip(".") +
            (". " + cultural_suffix if cultural_suffix else "")
        ).strip()
        reasoning_trace = "No strong cultural signals detected."

    # ── Model comparison text ──────────────────────────────────
    if compare_models:
        comparison_text = (
            f"ESN → {results['esn']['mode']} ({esn_conf:.2f})   |   "
            f"GRU → {results['gru']['mode']} ({gru_conf:.2f})\n"
            f"{esn_explain if esn_explain else 'ESN: weak activation'}"
        )
    else:
        comparison_text = esn_explain if esn_explain else ""

    # ── Translation ────────────────────────────────────────────
    translated = translate_to_language(final_caption_en, language)
    local_caption = (
        naturalize_translation(translated, language)
        if language != "English"
        else translated
    )

    # ── Hashtags ───────────────────────────────────────────────
    hashtags_str = generate_hashtags(final_caption_en, min_tags=2)

    # ── Cultural entity extraction ──────────────────────────────
    signal_vector = extract_signal_vector(base_caption)
    signal_summary = summarize_signals(signal_vector)

    cultural_index = build_cultural_index(signal_vector)

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO cultural_explorer (
                mode, confidence, english_caption, local_caption,
                hashtags, signal_vector, signal_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mode,
            round(overall_conf, 3),
            final_caption_en,
            local_caption,
            hashtags_str,
            json.dumps(signal_vector),
            signal_summary
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Auto-save failed: {e}")


    # ── Audio narration ────────────────────────────────────────
    audio_file = None
    try:
        audio_path = OUT_AUDIO / "caption.mp3"
        tts_text = local_caption if local_caption.strip() else final_caption_en
        gTTS(tts_text, lang=TTS_LANG.get(language, "en")).save(audio_path)
        audio_file = str(audio_path)
    except Exception as e:
        print(f"[TTS] Failed: {e}")

    # ── VISUAL ANALYTICS (ADDED FEATURES) ──────────────────────
    try:
        cultural_signal_plot = build_cultural_signal_plot(base_caption)
    except Exception as e:
        print(f"[Plot] Failed: {e}")
        cultural_signal_plot = None

    try:
        cultural_tag_cloud = build_cultural_tag_cloud(base_caption)
    except Exception as e:
        print(f"[Tags] Failed: {e}")
        cultural_tag_cloud = "Could not generate cultural tags."

    # ── Context state (used by Q&A / RAG) ──────────────────────
    context_state = (
        final_caption_en + (" " + cultural_suffix if cultural_suffix else "")
    )
    disagreement_report = analyze_esn_gru_disagreement(results)

    # ── Return ALL outputs ─────────────────────────────────────
    return (
        final_caption_en,
        local_caption,
        hashtags_str,
        audio_file,
        str(image_path),
        context_state,
        comparison_text,
        reasoning_trace,
        results,
        cultural_signal_plot,
        cultural_tag_cloud,
        cultural_index,
        signal_vector,          # ← THIS
        disagreement_report,
        surprise_report,
        silence_report
    )


def process_qa(question, audio_input, history, image_path, context):
    if audio_input:
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_input) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source)

            question = recognizer.recognize_google(audio_data)

        except sr.UnknownValueError:
            question = ""

        except Exception as e:
            print("[Speech error]", e)
            question = ""


    if not image_path:
        return history + [["", "Please generate a caption first."]], None, None

    if not question:
        return history + [["", "I couldn't understand the audio. Please try speaking again or type your question."]], None, image_path

    context_trim = context[:800]
    system_prompt = f"""
        RULES (STRICT):
        - Respond directly with the answer.
        - Do NOT mention the image, the caption, or how you know the information.
        - Do NOT explain your role or reasoning process.
        - Write naturally, as if speaking to a curious person.
        - Be clear, grounded, and culturally accurate.

        Context to ground your answer:
        {context_trim}
        """

    try:
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
        answer = ollama_generate(prompt, temperature=0.75, max_tokens=700)
    except Exception as e:
        answer = f"Sorry, could not get answer from model: {str(e)}"

    answer = answer.capitalize().strip()
    if answer and not answer.endswith(('.', '!', '?')):
        answer += '.'

    audio_path = OUT_AUDIO / "answer.mp3"
    audio_file = None
    try:
        gTTS(answer, lang="en").save(audio_path)
        audio_file = str(audio_path)
    except:
        pass

    history.append([question, answer])
    return history, audio_file, image_path


def cultural_rag(prediction, caption, region="India-wide"):
    mode = extract_mode_from_prediction(prediction)
    facts_md = get_cultural_facts(mode)

    prompt = f"""
You are narrating a short cultural passage.

RULES (STRICT):
- Output ONLY the story text.
- Do NOT explain what you are doing.
- Do NOT mention "caption", "image", "facts", "database", or "AI".
- Do NOT introduce or conclude the passage.
- Write in natural, flowing prose (no headings, no bullets).
- Length: 3–5 sentences.

Context to ground the story:
{caption}

Cultural memory (use only if relevant):
{facts_md}

Region lens:
{region}

Task:
Write a culturally grounded narrative that naturally connects the scene to Indian heritage,
daily life, rituals, or traditions. Be respectful, vivid, and concrete.
"""

    try:
        story = ollama_generate(
            prompt,
            temperature=0.6,
            max_tokens=420
        )
        return facts_md, clean_llm_caption(story), "SQLite DB + Ollama"
    except Exception as e:
        return facts_md, f"Could not generate story: {str(e)}", "N/A"

def clean_llm_caption(text: str) -> str:
    """
    Removes common LLM self-referential or explanatory prefixes/suffixes.
    """
    if not text:
        return text

    BAD_PREFIXES = [
        "here is",
        "here’s",
        "this is",
        "below is",
        "the following",
        "rewritten version",
        "as requested",
        "in conclusion",
        "overall,",
    ]

    lines = text.strip().splitlines()

    # Keep only lines that look like actual content
    clean_lines = []
    for line in lines:
        l = line.strip().lower()
        if any(l.startswith(p) for p in BAD_PREFIXES):
            continue
        if l.startswith("note:") or l.startswith("explanation"):
            continue
        clean_lines.append(line.strip())

    cleaned = " ".join(clean_lines).strip()

    # Final polish
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'

    return cleaned


def regional_adapt(caption, target_culture):
    if not caption or not target_culture:
        return "Please generate a caption first and select a culture."

    prompt = f"""
You are rewriting an image caption.

RULES (IMPORTANT):
- Output ONLY the rewritten caption text.
- Do NOT explain what you are doing.
- Do NOT introduce yourself.
- Do NOT add headings, quotes, or commentary.
- Do NOT mention the word "caption".
- Write in one concise paragraph.

Task:
Rewrite the text below as if naturally described by someone from {target_culture}.
Preserve meaning, adapt tone, phrasing, and cultural nuance.

Text:
{caption}
"""

    try:
        adapted = ollama_generate(
            prompt,
            temperature=0.65,
            max_tokens=220
        )
        return clean_llm_caption(adapted)
    except Exception as e:
        return f"[Error adapting caption] {str(e)}"


def accessibility_description(caption):
    prompt = f"""
        RULES (STRICT):
        - Write ONLY the description itself.
        - Do NOT mention images, captions, photos, or descriptions.
        - Do NOT explain what you are doing.
        - Use short, concrete sentences.
        - Describe objects, layout, actions, and atmosphere.
        - Avoid assumptions and cultural speculation unless clearly indicated.
        - If something is unclear, state it neutrally.

        Description:
        {caption}
        """

    try:
        response = ollama_generate(prompt, temperature=0.5, max_tokens=500)
    except:
        response = "Could not generate accessibility description."

    audio_path = OUT_AUDIO / "accessibility.mp3"
    audio_file = None
    try:
        gTTS(response, lang="en").save(audio_path)
        audio_file = str(audio_path)
    except:
        pass

    return response, audio_file

from datetime import datetime

def temporal_weight(added_at: str) -> float:
    """
    Exponential temporal decay.
    Handles SQLite naive timestamps safely.
    """
    try:
        t = datetime.fromisoformat(added_at)

        # 🔑 FIX: force UTC if SQLite timestamp is naive
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

    except Exception:
        return 1.0

    now = datetime.now(timezone.utc)
    age_days = (now - t).total_seconds() / 86400
    return float(np.exp(-TEMPORAL_DECAY_LAMBDA * age_days))

def generate_heritage_graph(_, mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT signal_vector, added_at
        FROM cultural_explorer
        WHERE mode = ?
        ORDER BY added_at DESC
        LIMIT 80
    """, (mode,))
    rows = c.fetchall()
    conn.close()

    G = nx.Graph()

    for signal_json, added_at in rows:
        if not signal_json:
            continue

        try:
            signals = json.loads(signal_json)
        except Exception:
            continue

        w = temporal_weight(added_at)

        active = [k for k, v in signals.items() if v >= 0.65]

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]

                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                else:
                    G.add_edge(a, b, weight=w)

    if not G.nodes:
        G.add_node("No dominant cultural signals yet")

    # ── Visualization ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(G, weight="weight", k=1.1, seed=42)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges]

    # ── Community detection ─────────────────────────
    communities = list(greedy_modularity_communities(G, weight="weight"))

    node_color_map = {}
    for idx, comm in enumerate(communities):
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        for node in comm:
            node_color_map[node] = color

    node_colors = [node_color_map.get(n, "#cccccc") for n in G.nodes]

    # ── Draw clustered graph ────────────────────────
    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=2600,
        node_color=node_colors,
        edge_color="#999999",
        width=[max(0.6, w) for w in edge_weights],
        font_size=10,
        font_weight="bold",
        alpha=0.92
    )


    return fig, [
        {
            "signal_a": u,
            "signal_b": v,
            "temporal_weight": round(G[u][v]["weight"], 3)
        }
        for u, v in G.edges
    ]

def generate_heritage_evolution_snapshot(mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT signal_vector
        FROM cultural_explorer
        WHERE mode = ?
        ORDER BY added_at DESC
        LIMIT 50
    """, (mode,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cultural history yet",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    G = nx.Graph()

    for (signal_json,) in rows:
        signals = json.loads(signal_json)
        active = [k for k, v in signals.items() if v >= 0.65]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                G.add_edge(active[i], active[j])

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos, ax=ax,
        node_color="#f4a261",
        node_size=2400,
        font_size=10,
        with_labels=True
    )

    ax.set_title("Cultural Signal Co-evolution")
    return fig


def safety_metrics(prediction):
    if not prediction:
        return 1.0, 0.0, 0.0

    mode = extract_mode_from_prediction(prediction)
    conf = max(
        prediction["esn"]["confidence"],
        prediction["gru"]["confidence"]
    )

    hallucination_risk = round(1 - conf, 2)
    vision_align = round(0.85 if mode != "generic" else 0.65, 2)
    cultural_conf = round(conf, 2)

    return hallucination_risk, vision_align, cultural_conf


def model_brain(results):
    if results is None or not isinstance(results, dict):
        return (
            "No model prediction available yet",
            "No model prediction available yet",
            "Upload and caption an image first"
        )

    esn = results.get("esn", {})
    gru = results.get("gru", {})

    esn_text = f"ESN → {esn.get('mode', 'N/A')} (confidence: {esn.get('confidence', 'N/A'):.3f})"
    gru_text = f"GRU → {gru.get('mode', 'N/A')} (confidence: {gru.get('confidence', 'N/A'):.3f})"

    return esn_text, gru_text, ""

def analyze_esn_gru_disagreement(results: dict) -> dict:
    if not results or "esn" not in results or "gru" not in results:
        return {"status": "No prediction available"}

    esn = results["esn"]
    gru = results["gru"]

    disagreement = esn["mode"] != gru["mode"]

    return {
        "disagreement": disagreement,
        "esn_mode": esn["mode"],
        "gru_mode": gru["mode"],
        "esn_confidence": round(esn["confidence"], 3),
        "gru_confidence": round(gru["confidence"], 3),
        "confidence_gap": round(abs(esn["confidence"] - gru["confidence"]), 3),
        "note": (
            "Models agree → stable cultural interpretation"
            if not disagreement else
            "Models disagree → ambiguous cultural cues"
        )
    }

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
with gr.Blocks(
    title="Cultural AI Explorer • Local Ollama",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="stone",
        radius_size="lg",
        font=["Inter", "sans-serif"]
    ),
    css="""
    .app-title { text-align: center; padding: 2rem 0 1rem; }
    .app-title h1 { font-size: 2.6rem; font-weight: 700; color: #d35400; margin-bottom: 0.3rem; }
    .app-title p { font-size: 1.1rem; color: #7f8c8d; }
    .card { background: #ffffff; border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.06); }
    """
) as demo:

    gr.HTML("""
    <div class="app-title">
        <h1>🧠 Cultural AI Explorer • Local Ollama</h1>
        <p>Offline • Explainable • Culturally Grounded • SQLite + Ollama</p>
    </div>
    """)

    with gr.Tabs():

        # =========================================================
        # Caption Generation
        # =========================================================
        with gr.Tab("Caption Generation"):
            with gr.Row(equal_height=True):

                # -------- Left column
                with gr.Column(scale=1, min_width=360):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📤 Upload & Settings")
                        img = gr.Image(type="pil", label="Upload Image", height=360)
                        lang = gr.Dropdown(
                            choices=list(TTS_LANG.keys()),
                            value="English",
                            label="Output Language"
                        )
                        use_esn = gr.Checkbox(
                            value=True,
                            label="Inject Cultural Context (ESN)"
                        )
                        compare_models = gr.Checkbox(
                            value=False,
                            label="Compare ESN vs GRU"
                        )
                        btn = gr.Button(
                            "✨ Generate Caption",
                            variant="primary",
                            size="lg"
                        )

                # -------- Right column
                with gr.Column(scale=2, min_width=520):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📝 Generated Output")

                        out_en = gr.Textbox(
                            label="Final Caption (English)",
                            lines=3,
                            interactive=False,
                            show_copy_button=True
                        )
                        out_local = gr.Textbox(
                            label="Translated Caption",
                            lines=3,
                            interactive=False,
                            show_copy_button=True
                        )
                        out_tags = gr.Textbox(
                            label="Suggested Hashtags",
                            lines=2,
                            interactive=False,
                            show_copy_button=True
                        )

                        comparison_out = gr.Textbox(
                            label="Model Comparison (ESN vs GRU)",
                            lines=2,
                            interactive=False
                        )
                        reasoning_out = gr.Textbox(
                            label="Reasoning Trace",
                            lines=3,
                            interactive=False
                        )

                        gr.Markdown("### 🔊 Audio Narration")
                        audio = gr.Audio(
                            type="filepath",
                            label="Listen to Caption"
                        )

                        # ======= ADDED VISUAL ANALYTICS =======
                        gr.Markdown("### 📊 Cultural Signal Analysis")
                        cultural_signal_plot = gr.Plot(
                            label="Cultural Signal Strength (ESN)"
                        )

                        gr.Markdown("### 🏷️ Cultural Tag Cloud")
                        cultural_tag_cloud = gr.Textbox(
                            label="Detected Cultural Tags",
                            interactive=False
                        )
                        # ====================================

            # -------- States
            image_state = gr.State(None)
            context_state = gr.State(None)
            prediction_state = gr.State(None)
            archive_index = gr.JSON(label="Cultural Signal Index")
            disagreement_view = gr.JSON(label="Disagreement Analysis")
            surprise_json = gr.JSON(label="Surprise Analysis")
            silence_json = gr.JSON(label="Silence Analysis")
            signal_state = gr.State({})

            # -------- Button wiring (FIXED)
            btn.click(
                fn=run_caption,
                inputs=[img, lang, use_esn, compare_models],
                outputs=[
                    out_en,
                    out_local,
                    out_tags,
                    audio,
                    image_state,
                    context_state,
                    comparison_out,
                    reasoning_out,
                    prediction_state,
                    cultural_signal_plot,
                    cultural_tag_cloud,
                    archive_index,
                    signal_state,           # ← THIS
                    disagreement_view,
                    surprise_json,
                    silence_json
                ]
            )

        # =========================================================
        # Q&A Chatbot
        # =========================================================
        with gr.Tab("Q&A Chatbot"):
            with gr.Row():
                with gr.Column():
                    original_media_qa = gr.Image(
                        label="Uploaded Media",
                        interactive=False,
                        height=200,
                        type="filepath"
                    )
                    chatbot = gr.Chatbot(label="Chat with the Image")
                    text_input = gr.Textbox(label="Type your question")
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Or speak your question",
                        format="wav"
                    )

                    qa_btn = gr.Button("Send")

                    qa_audio = gr.Audio(
                        type="filepath",
                        label="Listen to Answer"
                    )

                    qa_btn.click(
                        fn=process_qa,
                        inputs=[
                            text_input,
                            mic_input,
                            chatbot,
                            image_state,
                            context_state
                        ],
                        outputs=[
                            chatbot,
                            qa_audio,
                            original_media_qa
                        ]
                    )

        # =========================================================
        # Cultural Memory (RAG)
        # =========================================================
        with gr.Tab("Cultural Memory (RAG)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🌟 Cultural Facts & Story")
                    cultural_facts = gr.Markdown()
                    cultural_story = gr.Textbox(
                        label="Cultural Story",
                        lines=5
                    )
                    cultural_refs = gr.Textbox(
                        label="Sources / Knowledge",
                        lines=3
                    )

            refresh_rag = gr.Button("Generate Cultural Memory")

            refresh_rag.click(
                fn=cultural_rag,
                inputs=[
                    prediction_state,
                    context_state,
                    gr.State("India-wide")
                ],
                outputs=[
                    cultural_facts,
                    cultural_story,
                    cultural_refs
                ]
            )

        # =========================================================
        # Trust & Safety
        # =========================================================
        with gr.Tab("Trust & Safety"):
            with gr.Row():
                bias_score = gr.Slider(0, 1, label="Cultural Hallucination Risk")
                vision_score = gr.Slider(0, 1, label="Vision-Text Consistency")
                cultural_conf = gr.Slider(0, 1, label="Cultural Confidence")

            safety_btn = gr.Button("Compute Safety Metrics")

            safety_btn.click(
                fn=safety_metrics,
                inputs=[
                    prediction_state
                ],
                outputs=[
                    bias_score,
                    vision_score,
                    cultural_conf
                ]
            )

        # =========================================================
        # Model Brain
        # =========================================================
        with gr.Tab("Model Brain"):
            with gr.Row():
                esn_state = gr.Textbox(
                    label="ESN Cultural Activation",
                    lines=5
                )
                gru_state = gr.Textbox(
                    label="GRU Activation",
                    lines=5
                )
                trigger_words = gr.Textbox(
                    label="Cultural Triggers",
                    lines=3
                )

            brain_btn = gr.Button("Show Model Brain")

            brain_btn.click(
                fn=model_brain,
                inputs=[prediction_state],
                outputs=[
                    esn_state,
                    gru_state,
                    trigger_words
                ]
            )

        with gr.Tab("Cultural Evolution"):
            gr.Markdown("### ⏳ Cultural Signal Evolution Over Time")

            evolution_plot = gr.Plot()
            evolution_btn = gr.Button("Animate Cultural Evolution")

            evolution_btn.click(
                fn=lambda p: generate_heritage_evolution_snapshot(extract_mode_from_prediction(p)),
                inputs=[prediction_state],
                outputs=evolution_plot
            )

        # =========================================================
        # Regional Culture
        # =========================================================
        with gr.Tab("Regional Culture"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Caption")
                    original_reg = gr.Textbox(
                        label="Original (as generated)",
                        lines=4,
                        interactive=False,
                        value="Generate a caption first →"
                    )

                with gr.Column():
                    gr.Markdown("### Adapted to Selected Culture")
                    regional_caption = gr.Textbox(
                        label="Adapted version",
                        lines=6,
                        interactive=False
                    )

            with gr.Row():
                region = gr.Dropdown(
                    choices=[
                        "Japan", "Italy", "Mexico", "France", "Thailand", "Morocco",
                        "Brazil", "Korea", "Turkey", "Lebanon", "Ethiopia", "Vietnam",
                        "Peru", "Greece", "Spain", "Nigeria", "Indonesia", "Sweden",
                        "United States (Southern)",
                        "United States (New York style)",
                        "Generic international fusion"
                    ],
                    label="Select Target Culture / Country",
                    value="Japan"
                )
                regional_btn = gr.Button(
                    "Adapt to This Culture",
                    variant="primary"
                )

            regional_btn.click(
                fn=lambda cap, cult: (cap, regional_adapt(cap, cult)),
                inputs=[context_state, region],
                outputs=[original_reg, regional_caption]
            )

        # with gr.Tab("🧠 Model Cognition (3D)"):
        #     gr.Markdown("### ESN & GRU Internal Cognition Space")

        #     gr.HTML("""
        #     <iframe src="/file=static/esn_evolution.html" width="100%" height="520"></iframe>
        #     <iframe src="/file=static/esn_vs_gru.html" width="100%" height="520"></iframe>
        #     <iframe src="/file=static/counterfactual_drift.html" width="100%" height="520"></iframe>
        #     """)

        # =========================================================
        # Accessibility
        # =========================================================
        with gr.Tab("Accessibility"):
            with gr.Row():
                blind_caption = gr.Textbox(
                    label="Screen Reader Friendly Description",
                    lines=5
                )
                blind_audio = gr.Audio(
                    label="Audio Narration for Blind Users"
                )

            accessibility_btn = gr.Button("Generate Accessibility Mode")

            accessibility_btn.click(
                fn=accessibility_description,
                inputs=context_state,
                outputs=[blind_caption, blind_audio]
            )
            
        # Cultural Heritage Map
        # =========================================================
        with gr.Tab("Cultural Heritage Map"):
            with gr.Row():
                heritage_graph = gr.Plot(
                    label="Heritage Graph Visualization"
                )
                heritage_out = gr.JSON(
                    label="Heritage Data"
                )

            map_btn = gr.Button("Generate Heritage Map")

            map_btn.click(
                fn=lambda _, p: generate_heritage_graph(_, extract_mode_from_prediction(p)),
                inputs=[context_state, prediction_state],
                outputs=[heritage_graph, heritage_out]
            )
    
        with gr.Tab("What-If Cultural Frame"):
            frame = gr.Dropdown(
                choices=list(COUNTERFACTUAL_FRAMES.keys()),
                label="Re-interpret cultural memory"
            )
            cf_out = gr.JSON(label="Counterfactual Interpretation")

            frame.change(
                fn=lambda s, f: counterfactual_from_vector(s, f),
                inputs=[signal_state, frame],
                outputs=cf_out
            )

        # =========================================================
        # Similar Images
        # =========================================================
        with gr.Tab("Similar Images"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Find Similar Images Online")
                    similar_gallery = gr.Gallery(
                        label="Similar Images",
                        columns=4,
                        height="auto",
                        object_fit="contain"
                    )
                    refresh_btn = gr.Button("Refresh Similar Images")

            refresh_btn.click(
                fn=search_similar_images,
                inputs=context_state,
                outputs=similar_gallery
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
