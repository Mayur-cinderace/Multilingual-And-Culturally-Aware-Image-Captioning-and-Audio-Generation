# app.py – Cultural AI Explorer with SQLite DB + Ollama (ENHANCED UI)
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

# Import visualization if available
try:
    from esn_gru_visualization import (
        animate_esn_reservoir,
        compare_esn_gru_3d,
        counterfactual_drift_animation
    )
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False
    print("[Warning] esn_gru_visualization not available - 3D visualizations disabled")

load_dotenv()

# ────────────────────────────────────────────────
#   LOCAL OLLAMA CLIENT
# ────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

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
            signal_vector TEXT,
            signal_summary TEXT,
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

# ── IMPORT ENHANCED MODULES ──
from trust_safety_enhanced import CulturalTrustAnalyzer, ModelBrainAnalyzer
from cultural_analysis_enhanced import (
    CulturalSignalAnalyzer,
    ModelDisagreementAnalyzer,
    CulturalSurpriseDetector,
    CulturalSilenceDetector,
    CounterfactualFrameAnalyzer
)

# ── INITIALIZE ANALYZERS ──
trust_analyzer = CulturalTrustAnalyzer()
brain_analyzer = ModelBrainAnalyzer()
signal_analyzer = CulturalSignalAnalyzer()
disagreement_analyzer = ModelDisagreementAnalyzer()
surprise_detector = CulturalSurpriseDetector()
silence_detector = CulturalSilenceDetector()
counterfactual_analyzer = CounterfactualFrameAnalyzer()

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

    # daily_life  ── 4 examples
    ("Family having a simple home meal.", "daily_life"),
    ("Daily routine with breakfast in bowl.", "daily_life"),
    ("Home-cooked dish served to family.", "daily_life"),
    ("Everyday gathering around the table.", "daily_life"),

    # generic  ── 18 examples
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

        for t in templates:
            augmented.append((t.format(caption.lower()), mode))

        words = caption.split()
        if len(words) > 5:
            truncated = " ".join(words[:len(words)//2])
            augmented.append((truncated, mode))

        augmented.append((caption + noise_suffixes[np.random.randint(len(noise_suffixes))], mode))

    return augmented


cultural_manager = CulturalContextManager()
augmented_samples = augment_samples(training_samples)
cultural_manager.train(augmented_samples)

CULTURAL_CONFIDENCE_THRESHOLD = 0.55

# ── IMPROVED CULTURAL TEMPLATES ──
CULTURAL_TEMPLATES = {
    "food_traditional": {
        "prefix": "traditional ",
        "suffix": " This reflects regional cooking traditions with distinctive spice blends and serving styles."
    },
    "festival_context": {
        "prefix": "festive ",
        "suffix": " Such scenes are central to cultural celebrations and communal gatherings."
    },
    "daily_life": {
        "prefix": "everyday ",
        "suffix": " This captures familiar routines and family moments."
    },
    "generic": {
        "prefix": "",
        "suffix": ""
    }
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

TEMPORAL_DECAY_LAMBDA = 0.12


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
    """Use the enhanced counterfactual analyzer"""
    if not signal_vector:
        return {"error": "No cultural signal available yet."}

    modifiers = COUNTERFACTUAL_FRAMES.get(frame, {})
    return counterfactual_analyzer.analyze_shift(signal_vector, frame, modifiers)


def build_cultural_signal_plot(caption: str):
    """Enhanced with signal analyzer"""
    analysis = signal_analyzer.analyze_signal_strength(caption, cultural_manager.esn)
    
    dimensions = analysis.get("dimensions", {})
    if not dimensions:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No cultural signals detected", ha="center", va="center")
        ax.axis("off")
        return fig
    
    labels = list(dimensions.keys())
    values = [d["strength"] for d in dimensions.values()]
    levels = [d["level"] for d in dimensions.values()]
    
    # Color code by level
    colors = []
    for level in levels:
        if level == "Strong":
            colors.append("#2a9d8f")
        elif level == "Moderate":
            colors.append("#f4a261")
        else:
            colors.append("#e63946")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Activation Strength")
    ax.set_title("Cultural Signal Strength Analysis")
    ax.set_xlim(0, 1.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2a9d8f', label='Strong'),
        Patch(facecolor='#f4a261', label='Moderate'),
        Patch(facecolor='#e63946', label='Weak')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig

def build_cultural_tag_cloud(caption: str):
    """Enhanced with signal analyzer"""
    analysis = signal_analyzer.analyze_signal_strength(caption, cultural_manager.esn)
    
    dominant_signals = analysis.get("dominant_signals", [])
    if not dominant_signals:
        return "No dominant cultural tags detected."
    
    tags = [f"{s['name']} ({s['strength']:.2f})" for s in dominant_signals]
    return " • ".join(tags)


def build_cultural_index(signal_vector: dict) -> dict:
    index = {}
    for k, v in signal_vector.items():
        index[k] = round(min(1.0, max(0.0, float(v))), 3)
    return index

def extract_mode_from_prediction(pred):
    if not pred or not isinstance(pred, dict):
        return "generic"
    return pred.get("combined_mode", "generic")

def cultural_entropy(features: np.ndarray, eps=1e-9) -> float:
    x = np.abs(features.astype(float))
    x = x / (np.sum(x) + eps)
    entropy = -np.sum(x * np.log(x + eps))
    return float(entropy)

def detect_cultural_surprise(caption: str, prediction: dict) -> dict:
    """Use the enhanced surprise detector"""
    return surprise_detector.detect(caption, prediction, cultural_manager.esn)


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
# ── ENHANCED: ESN/GRU CULTURAL INJECTION LOGIC ──
# ------------------------------------------------------------
def get_cultural_injection(caption: str, mode: str, esn_conf: float, gru_conf: float, top_k=3) -> tuple[str, str, str]:
    """
    Returns comprehensive cultural enhancement based on BOTH models.
    
    Returns:
    - prefix_injection: text to prepend
    - suffix_context: cultural context to append
    - reasoning_trace: explanation of what activated
    """
    
    # No injection for weak predictions
    if max(esn_conf, gru_conf) < CULTURAL_CONFIDENCE_THRESHOLD:
        return "", "", "Confidence too low for cultural injection"
    
    # Get ESN feature activations
    feats = cultural_manager.esn.extract_features(caption)
    feature_names = [
        "caption_length", "spice_level", "ritual_score", "daily_score", "heritage",
        "color_vivid", "texture_words", "serving_style", "has_thali", "has_festival",
        "has_sweet", "has_rice", "veg_focus", "meat_focus", "has_dessert", "has_beverage",
        "has_offering", "has_decoration", "has_people", "has_utensil",
        "adj_ratio", "uniq_ratio", "sentiment_pos"
    ]
    
    # Get top activated features (excluding bias terms)
    top_indices = np.argsort(feats[:23])[::-1][:top_k]
    top_features = [(feature_names[i], round(float(feats[i]), 2)) for i in top_indices if feats[i] > 0.4]
    
    # Build reasoning trace
    reasoning_parts = []
    reasoning_parts.append(f"ESN → {mode} ({esn_conf:.3f})")
    reasoning_parts.append(f"GRU → {mode} ({gru_conf:.3f})")
    if top_features:
        reasoning_parts.append(f"Top signals: {', '.join(f'{n}={v}' for n, v in top_features[:3])}")
    
    reasoning_trace = " | ".join(reasoning_parts)
    
    # Get template for this mode
    template = CULTURAL_TEMPLATES.get(mode, CULTURAL_TEMPLATES["generic"])
    prefix = template["prefix"]
    suffix = template["suffix"]
    
    # Additional specific injections based on features
    if mode == "food_traditional":
        if feats[8] > 1.4:  # has_thali
            prefix = "traditional thali of "
        elif feats[1] > 1.1:  # spice_level
            prefix = "richly spiced "
        elif feats[4] > 0.9:  # heritage
            prefix = "heritage-style "
    
    elif mode == "festival_context":
        if feats[9] > 1.5:  # has_festival
            prefix = "festive ceremonial "
        elif feats[16] > 1.2:  # has_offering
            prefix = "ritual offering of "
    
    elif mode == "daily_life":
        if feats[3] > 0.9:  # daily_score
            prefix = "everyday "
        elif feats[18] > 0.8:  # has_people
            prefix = "family gathering with "
    
    return prefix, suffix, reasoning_trace


import re
from collections import Counter

STOPWORDS = {
    "the","a","an","with","and","or","of","in","on","at","to","for","from",
    "this","that","is","are","was","were","shows","showing","image","picture",
    "reflects","featuring","often","style","styles","scene","view"
}

def build_pexels_query(caption: str, max_terms: int = 5) -> str:
    if not caption:
        return "photo"

    text = caption.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 3]

    if not tokens:
        return "photo"

    freq = Counter(tokens)
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], tokens.index(x[0])))
    keywords = [w for w, _ in sorted_tokens[:max_terms]]

    return " ".join(keywords)

def detect_cultural_silence(caption: str) -> dict:
    """Use the enhanced silence detector"""
    return silence_detector.detect(caption, cultural_manager.esn)


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
# ── ENHANCED: Core Caption Function ──
# ------------------------------------------------------------
def run_caption(image, language, use_esn, compare_models=False):
    """
    Enhanced version with comprehensive trust & safety analysis.
    """
    if image is None:
        return (
            "", "", "", None, None, None, "", "No image provided.",
            None, None, "", {}, {}, {}, {}, {}, {}, {}
        )

    # Save image
    image_path = "temp.jpg"
    image.save(image_path)

    # Base caption (vision-grounded)
    base_caption = generate_base_caption(image_path).strip()

    # Cultural classification
    results = cultural_manager.predict(base_caption)
    
    # 3D Visualizations (if available)
    if VISUALIZATIONS_AVAILABLE:
        try:
            animate_esn_reservoir(
                base_caption,
                cultural_manager.esn,
                output_path="static/esn_evolution.html"
            )
            compare_esn_gru_3d(
                base_caption,
                cultural_manager.esn,
                cultural_manager.gru,
                output_path="static/esn_vs_gru.html"
            )
            counterfactual_drift_animation(
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
    
    # ── ENHANCED: Comprehensive Analysis ──
    signal_vector = extract_signal_vector(base_caption)
    
    # Comprehensive trust analysis
    trust_report = trust_analyzer.analyze_comprehensive(
        results, base_caption, signal_vector, cultural_manager.esn
    )
    
    # Model brain analysis
    brain_report = brain_analyzer.analyze_model_cognition(
        results, base_caption, cultural_manager.esn, cultural_manager.gru
    )
    
    # Signal analysis
    signal_analysis = signal_analyzer.analyze_signal_strength(
        base_caption, cultural_manager.esn
    )
    
    # Disagreement analysis
    disagreement_report = disagreement_analyzer.analyze(
        results, base_caption, cultural_manager.esn, cultural_manager.gru
    )
    
    # Surprise detection
    surprise_report = surprise_detector.detect(
        base_caption, results, cultural_manager.esn
    )
    
    # Silence detection
    silence_report = silence_detector.detect(
        base_caption, cultural_manager.esn
    )
    
    # ── Cultural injection ──
    prefix_inject = ""
    suffix_context = ""
    reasoning_trace = ""
    
    if use_esn:
        prefix_inject, suffix_context, reasoning_trace = get_cultural_injection(
            base_caption, mode, esn_conf, gru_conf
        )
        prefix_inject = ""
    
    # Build final caption with injection
    final_caption_en = base_caption
    
    if prefix_inject:
        words = base_caption.split()
        if len(words) > 0:
            final_caption_en = f"{prefix_inject}{base_caption}"
    
    if suffix_context:
        final_caption_en = final_caption_en.rstrip('.') + '. ' + suffix_context.strip()
    
    final_caption_en = final_caption_en.strip()
    
    # Model comparison text
    if compare_models:
        comparison_text = (
            f"ESN → {results['esn']['mode']} ({esn_conf:.2f})   |   "
            f"GRU → {results['gru']['mode']} ({gru_conf:.2f})\n"
            f"{reasoning_trace}"
        )
    else:
        comparison_text = reasoning_trace if reasoning_trace else ""

    # Translation
    translated = translate_to_language(final_caption_en, language)
    local_caption = (
        naturalize_translation(translated, language)
        if language != "English"
        else translated
    )

    # Enhanced hashtag generation
    hashtags_str = generate_hashtags(final_caption_en, min_tags=3, max_tags=5)

    signal_summary = summarize_signals(signal_vector)
    cultural_index = build_cultural_index(signal_vector)

    # Save to DB
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

    # Audio narration
    audio_file = None
    try:
        audio_path = OUT_AUDIO / "caption.mp3"
        tts_text = local_caption if local_caption.strip() else final_caption_en
        gTTS(tts_text, lang=TTS_LANG.get(language, "en")).save(audio_path)
        audio_file = str(audio_path)
    except Exception as e:
        print(f"[TTS] Failed: {e}")

    # Visual analytics
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

    context_state = final_caption_en

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
        signal_vector,
        disagreement_report,
        surprise_report,
        silence_report,
        trust_report,
        brain_report
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
        story = ollama_generate(prompt, temperature=0.6, max_tokens=420)
        return facts_md, clean_llm_caption(story), "SQLite DB + Ollama"
    except Exception as e:
        return facts_md, f"Could not generate story: {str(e)}", "N/A"

def clean_llm_caption(text: str) -> str:
    if not text:
        return text

    BAD_PREFIXES = [
        "here is", "here's", "this is", "below is", "the following",
        "rewritten version", "as requested", "in conclusion", "overall,",
    ]

    lines = text.strip().splitlines()
    clean_lines = []
    for line in lines:
        l = line.strip().lower()
        if any(l.startswith(p) for p in BAD_PREFIXES):
            continue
        if l.startswith("note:") or l.startswith("explanation"):
            continue
        clean_lines.append(line.strip())

    cleaned = " ".join(clean_lines).strip()

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
        adapted = ollama_generate(prompt, temperature=0.65, max_tokens=220)
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


def temporal_weight(added_at: str) -> float:
    """Exponential temporal decay with proper timezone handling."""
    try:
        t = datetime.fromisoformat(added_at)
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
        LIMIT 100
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
        active = [k for k, v in signals.items() if v >= 0.6]

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]

                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                else:
                    G.add_edge(a, b, weight=w)

    if not G.nodes:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No cultural signals yet - generate some captions first!",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig, []

    fig, ax = plt.subplots(figsize=(10, 7))
    
    pos = nx.spring_layout(G, weight="weight", k=1.5, iterations=50, seed=42)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    
    if edge_weights:
        max_weight = max(edge_weights)
        normalized_weights = [w / max_weight for w in edge_weights]
    else:
        normalized_weights = []

    communities = list(greedy_modularity_communities(G, weight="weight"))

    node_color_map = {}
    for idx, comm in enumerate(communities):
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        for node in comm:
            node_color_map[node] = color

    node_colors = [node_color_map.get(n, "#cccccc") for n in G.nodes]
    node_sizes = [2000 + 400 * G.degree(n) for n in G.nodes]

    nx.draw(
        G, pos,
        ax=ax,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="#999999",
        width=[max(0.5, 3 * w) for w in normalized_weights],
        font_size=9,
        font_weight="bold",
        alpha=0.9
    )

    ax.set_title(f"Cultural Heritage Network - {mode.replace('_', ' ').title()}", fontsize=14)
    plt.tight_layout()

    edge_data = [
        {
            "signal_a": u,
            "signal_b": v,
            "temporal_weight": round(G[u][v]["weight"], 3),
            "occurrences": int(G[u][v]["weight"] / 0.5)
        }
        for u, v in sorted(G.edges, key=lambda e: -G[e[0]][e[1]]["weight"])[:20]
    ]

    return fig, edge_data


def generate_heritage_evolution_snapshot(mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT signal_vector, added_at
        FROM cultural_explorer
        WHERE mode = ?
        ORDER BY added_at ASC
        LIMIT 100
    """, (mode,))
    rows = c.fetchall()
    conn.close()

    if not rows or len(rows) < 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data for evolution - need at least 3 entries",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    timestamps = []
    signal_strengths = {}
    
    for signal_json, added_at in rows:
        try:
            signals = json.loads(signal_json)
            t = datetime.fromisoformat(added_at)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            timestamps.append(t)
            
            for signal_name, strength in signals.items():
                if signal_name not in signal_strengths:
                    signal_strengths[signal_name] = []
                signal_strengths[signal_name].append(strength)
        except:
            continue
    
    if not timestamps:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid temporal data", ha="center", va="center")
        ax.axis("off")
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    start_time = min(timestamps)
    relative_days = [(t - start_time).total_seconds() / 86400 for t in timestamps]
    
    signal_means = {k: np.mean(v) for k, v in signal_strengths.items()}
    top_signals = sorted(signal_means.items(), key=lambda x: -x[1])[:4]
    
    colors_cycle = ['#e63946', '#457b9d', '#2a9d8f', '#f4a261']
    
    for idx, (signal_name, _) in enumerate(top_signals):
        values = signal_strengths[signal_name]
        
        if len(values) > 5:
            window = 5
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smoothed_days = relative_days[window-1:]
        else:
            smoothed = values
            smoothed_days = relative_days
        
        ax.plot(smoothed_days, smoothed, 
                marker='o', markersize=4,
                label=signal_name.replace('_', ' ').title(),
                color=colors_cycle[idx % len(colors_cycle)],
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Days Since First Entry", fontsize=11)
    ax.set_ylabel("Signal Strength", fontsize=11)
    ax.set_title(f"Cultural Signal Evolution - {mode.replace('_', ' ').title()}", fontsize=13, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def safety_metrics(prediction):
    """Enhanced safety metrics using trust analyzer"""
    if not prediction:
        return 1.0, 0.0, 0.0, {}

    # Get comprehensive trust analysis
    signal_vector = {}  # Will be populated from DB if needed
    trust_report = trust_analyzer.analyze_comprehensive(
        prediction, "", signal_vector, cultural_manager.esn
    )

    hallucination_risk = trust_report.get("hallucination_risk", {}).get("score", 0.5)
    vision_align = trust_report.get("vision_text_consistency", {}).get("score", 0.5)
    cultural_conf = trust_report.get("cultural_confidence", {}).get("score", 0.5)

    return hallucination_risk, vision_align, cultural_conf, trust_report


def model_brain(results, caption=""):
    """Enhanced model brain using brain analyzer"""
    if results is None or not isinstance(results, dict):
        return (
            "No model prediction available yet",
            "No model prediction available yet",
            "Upload and caption an image first",
            {}
        )

    brain_report = brain_analyzer.analyze_model_cognition(
        results, caption, cultural_manager.esn, cultural_manager.gru
    )

    esn_analysis = brain_report.get("esn_analysis", {})
    gru_analysis = brain_report.get("gru_analysis", {})
    
    esn_text = f"ESN → {esn_analysis.get('mode', 'N/A')} (confidence: {esn_analysis.get('confidence', 0):.3f})\n"
    esn_text += f"Cognitive Style: {esn_analysis.get('cognitive_style', 'N/A')}\n"
    esn_text += f"Top Features: {esn_analysis.get('top_features', [])}"
    
    gru_text = f"GRU → {gru_analysis.get('mode', 'N/A')} (confidence: {gru_analysis.get('confidence', 0):.3f})\n"
    gru_text += f"Cognitive Style: {gru_analysis.get('cognitive_style', 'N/A')}"
    
    comparison = brain_report.get("cognitive_comparison", {})
    trigger_text = comparison.get("interpretation", "")

    return esn_text, gru_text, trigger_text, brain_report

def analyze_esn_gru_disagreement(results: dict) -> dict:
    """Use the enhanced disagreement analyzer"""
    return disagreement_analyzer.analyze(
        results, "", cultural_manager.esn, cultural_manager.gru
    )

# ------------------------------------------------------------
# ENHANCED UI WITH MODERN DESIGN
# ------------------------------------------------------------

# Custom CSS for modern, professional look
custom_css = """
/* ═══════════════════════════════════════════════════════════════
   GLOBAL STYLES & VARIABLES
   ═══════════════════════════════════════════════════════════════ */
:root {
    --primary-gradient: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
    --secondary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent-color: #FF6B35;
    --text-primary: #2C3E50;
    --text-secondary: #7F8C8D;
    --bg-card: #FFFFFF;
    --bg-surface: #F8F9FA;
    --border-color: #E9ECEF;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.12);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --spacing-xs: 0.5rem;
    --spacing-sm: 1rem;
    --spacing-md: 1.5rem;
    --spacing-lg: 2rem;
}

/* ═══════════════════════════════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* ═══════════════════════════════════════════════════════════════
   HEADER & BRANDING
   ═══════════════════════════════════════════════════════════════ */
.app-header {
    background: var(--primary-gradient);
    padding: 3rem 2rem;
    margin: -1.5rem -1.5rem 2rem -1.5rem;
    text-align: center;
    border-radius: 0 0 32px 32px;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.app-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -5%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}

.app-title {
    position: relative;
    z-index: 1;
}

.app-title h1 {
    font-size: 3.2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 4px 12px rgba(0,0,0,0.15);
    letter-spacing: -0.03em;
}

.app-title .subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.95);
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}

.app-title .version-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
}

/* ═══════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════ */
.tabs {
    background: var(--bg-surface);
    border-radius: var(--radius-lg);
    padding: 0.5rem;
    margin-bottom: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
}

.tab-nav button {
    background: transparent;
    border: none;
    padding: 0.875rem 1.5rem;
    margin: 0 0.25rem;
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-secondary);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.tab-nav button:hover {
    background: rgba(255,107,53,0.08);
    color: var(--accent-color);
    transform: translateY(-1px);
}

.tab-nav button.selected {
    background: white;
    color: var(--accent-color);
    box-shadow: var(--shadow-sm);
}

.tab-nav button.selected::after {
    content: '';
    position: absolute;
    bottom: -0.5rem;
    left: 50%;
    transform: translateX(-50%);
    width: 40%;
    height: 3px;
    background: var(--primary-gradient);
    border-radius: 2px;
}

/* ═══════════════════════════════════════════════════════════════
   CARDS & CONTAINERS
   ═══════════════════════════════════════════════════════════════ */
.card, .gradio-container .gr-box {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.card-header {
    display: flex;
    align-items: center;
    margin-bottom: var(--spacing-md);
    padding-bottom: var(--spacing-sm);
    border-bottom: 2px solid var(--bg-surface);
}

.card-header h3 {
    font-size: 1.3rem;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--accent-color) 0%, transparent 100%);
    margin: var(--spacing-lg) 0;
    border-radius: 2px;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════ */
.gradio-container button {
    font-weight: 600;
    border-radius: var(--radius-md);
    padding: 0.875rem 1.75rem;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.gradio-container button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.gradio-container button:hover::before {
    width: 300px;
    height: 300px;
}

.gradio-container button.primary {
    background: var(--primary-gradient);
    color: white;
    box-shadow: 0 4px 14px rgba(255,107,53,0.35);
}

.gradio-container button.primary:hover {
    box-shadow: 0 6px 20px rgba(255,107,53,0.45);
    transform: translateY(-2px);
}

.gradio-container button.secondary {
    background: white;
    color: var(--accent-color);
    border: 2px solid var(--accent-color);
}

.gradio-container button.secondary:hover {
    background: var(--accent-color);
    color: white;
    transform: translateY(-2px);
}

/* ═══════════════════════════════════════════════════════════════
   INPUTS & TEXTBOXES
   ═══════════════════════════════════════════════════════════════ */
.gradio-container input,
.gradio-container textarea {
    border: 2px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 0.875rem 1rem;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    background: white;
}

.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(255,107,53,0.1);
    outline: none;
}

.gradio-container label {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
}

/* ═══════════════════════════════════════════════════════════════
   IMAGE UPLOAD
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .image-container {
    border: 3px dashed var(--border-color);
    border-radius: var(--radius-lg);
    transition: all 0.3s ease;
    background: var(--bg-surface);
}

.gradio-container .image-container:hover {
    border-color: var(--accent-color);
    background: rgba(255,107,53,0.03);
}

/* ═══════════════════════════════════════════════════════════════
   CHATBOT
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .chatbot {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
}

.gradio-container .message {
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    max-width: 80%;
}

.gradio-container .message.user {
    background: var(--primary-gradient);
    color: white;
    margin-left: auto;
}

.gradio-container .message.bot {
    background: var(--bg-surface);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

/* ═══════════════════════════════════════════════════════════════
   GALLERY
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .gallery {
    gap: 1rem;
}

.gradio-container .gallery-item {
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.gradio-container .gallery-item:hover {
    box-shadow: var(--shadow-md);
    transform: scale(1.03);
}

/* ═══════════════════════════════════════════════════════════════
   ACCORDION & COLLAPSIBLES
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .accordion {
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid var(--border-color);
}

.gradio-container .accordion-header {
    background: var(--bg-surface);
    padding: 1rem 1.5rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.gradio-container .accordion-header:hover {
    background: rgba(255,107,53,0.05);
}

/* ═══════════════════════════════════════════════════════════════
   PLOTS & VISUALIZATIONS
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .plot-container {
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    background: white;
    padding: 1rem;
}

/* ═══════════════════════════════════════════════════════════════
   JSON DISPLAY
   ═══════════════════════════════════════════════════════════════ */
.gradio-container .json-holder {
    background: #ffffff;
    border-radius: var(--radius-md);
    padding: 1.5rem;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.85rem;
    overflow-x: auto;
    box-shadow: var(--shadow-sm);
}

/* ═══════════════════════════════════════════════════════════════
   BADGES & TAGS
   ═══════════════════════════════════════════════════════════════ */
.badge {
    display: inline-block;
    padding: 0.35rem 0.875rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-primary {
    background: var(--primary-gradient);
    color: white;
}

.badge-success {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    color: white;
}

.badge-info {
    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
    color: white;
}

.badge-warning {
    background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
    color: white;
}

/* ═══════════════════════════════════════════════════════════════
   LOADING & ANIMATIONS
   ═══════════════════════════════════════════════════════════════ */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: slideIn 0.5s ease-out;
}

/* ═══════════════════════════════════════════════════════════════
   RESPONSIVE DESIGN
   ═══════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .app-title h1 {
        font-size: 2rem;
    }
    
    .app-header {
        padding: 2rem 1rem;
    }
    
    .card {
        padding: 1rem;
    }
    
    .tab-nav button {
        padding: 0.675rem 1rem;
        font-size: 0.85rem;
    }
}

/* ═══════════════════════════════════════════════════════════════
   UTILITY CLASSES
   ═══════════════════════════════════════════════════════════════ */
.text-center { text-align: center; }
.text-primary { color: var(--accent-color); }
.text-muted { color: var(--text-secondary); }
.mt-1 { margin-top: var(--spacing-xs); }
.mt-2 { margin-top: var(--spacing-sm); }
.mt-3 { margin-top: var(--spacing-md); }
.mb-1 { margin-bottom: var(--spacing-xs); }
.mb-2 { margin-bottom: var(--spacing-sm); }
.mb-3 { margin-bottom: var(--spacing-md); }
.p-0 { padding: 0; }
.p-1 { padding: var(--spacing-xs); }
.p-2 { padding: var(--spacing-sm); }
.p-3 { padding: var(--spacing-md); }

/* ═══════════════════════════════════════════════════════════════
   SCROLLBAR STYLING
   ═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-surface);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent-color), #F7931E);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #F7931E, var(--accent-color));
}
/* ═══════════════════════════════════════════════════════════════
   FIX: JSON VISIBILITY & SYNTAX HIGHLIGHTING
   ═══════════════════════════════════════════════════════════════ */

/* Main JSON container */
.gradio-container .json-holder,
.gradio-json {
    background: #f5f5f5 !important;
    color: #2c3e50 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    padding: 1.5rem !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

/* Dark mode variant */
.gradio-container.dark .json-holder,
.gradio-container.dark .gradio-json {
    background: #1e1e1e !important;
    color: #e0e0e0 !important;
}

/* Keys - make them visible */
.gradio-container .json-holder span.key,
.gradio-json span.key {
    color: #0066cc !important;
    font-weight: 600 !important;
}

/* Strings */
.gradio-container .json-holder span.string,
.gradio-json span.string {
    color: #008000 !important;
}

/* Numbers */
.gradio-container .json-holder span.number,
.gradio-json span.number {
    color: #e64980 !important;
    font-weight: 500 !important;
}

/* Booleans / null */
.gradio-container .json-holder span.boolean,
.gradio-container .json-holder span.null,
.gradio-json span.boolean,
.gradio-json span.null {
    color: #9966cc !important;
    font-weight: 600 !important;
}

/* Brackets & punctuation */
.gradio-container .json-holder span.punctuation,
.gradio-json span.punctuation {
    color: #2c3e50 !important;
    font-weight: 700 !important;
}

/* All text in JSON containers */
.gradio-container .json-holder *,
.gradio-json * {
    color: #2c3e50 !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* JSON container full width */
.gradio-container .json-holder,
.gradio-json {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Fix control panel height */
.controls-panel {
    min-height: 260px;
}

"""

with gr.Blocks(
    title="Cultural AI Explorer • Enhanced Trust & Safety",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="stone",
        radius_size="lg",
        font=["Inter", "sans-serif"],
        font_mono=["JetBrains Mono", "monospace"]
    ),
    css=custom_css
) as demo:

    # Enhanced Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">
            <h1>🧠 Cultural AI Explorer</h1>
            <p class="subtitle">Comprehensive Trust & Safety • Cultural Signal Analysis • Model Interpretability</p>
            <span class="version-badge">v2.0</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════
        # CAPTION GENERATION TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🎨 Caption Generation"):
            with gr.Row():
                # Left Column - Upload & Controls
                with gr.Column(scale=1, min_width=380):
                    gr.HTML('<div class="card-header"><h3>📤 Upload & Settings</h3></div>')
                    
                    img = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=380,
                        elem_classes="card"
                    )
                    
                    with gr.Group(elem_classes="card controls-panel"):
                        lang = gr.Dropdown(
                            choices=list(TTS_LANG.keys()),
                            value="English",
                            label="🌐 Output Language",
                            elem_classes="mb-2"
                        )
                        
                        use_esn = gr.Checkbox(
                            value=True,
                            label="🔮 Inject Cultural Context (ESN)",
                            elem_classes="mb-1"
                        )
                        
                        compare_models = gr.Checkbox(
                            value=False,
                            label="⚖️ Compare ESN vs GRU Models",
                            elem_classes="mb-2"
                        )
                        
                        btn = gr.Button(
                            "✨ Generate Caption",
                            variant="primary",
                            size="lg",
                            elem_classes="primary"
                        )

                # Right Column - Results
                with gr.Column(scale=2, min_width=600):
                    gr.HTML('<div class="card-header"><h3>📝 Generated Output</h3></div>')
                    
                    with gr.Group(elem_classes="card"):
                        out_en = gr.Textbox(
                            label="📄 Final Caption (English)",
                            lines=3,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes="mb-2"
                        )
                        
                        out_local = gr.Textbox(
                            label="🌏 Translated Caption",
                            lines=3,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes="mb-2"
                        )
                        
                        out_tags = gr.Textbox(
                            label="🏷️ Suggested Hashtags",
                            lines=2,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes="mb-2"
                        )
                    
                    gr.HTML('<div class="section-divider"></div>')
                    
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🔬 Model Analysis")
                        
                        comparison_out = gr.Textbox(
                            label="⚖️ Model Comparison (ESN vs GRU)",
                            lines=2,
                            interactive=False,
                            elem_classes="mb-2"
                        )
                        
                        reasoning_out = gr.Textbox(
                            label="🧩 Reasoning Trace",
                            lines=3,
                            interactive=False
                        )
                    
                    gr.HTML('<div class="section-divider"></div>')
                    
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🔊 Audio Narration")
                        audio = gr.Audio(
                            type="filepath",
                            label="Listen to Caption"
                        )
                    
                    gr.HTML('<div class="section-divider"></div>')
                    
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📊 Cultural Signal Analysis")
                        cultural_signal_plot = gr.Plot(
                            label="Cultural Signal Strength"
                        )
                        
                        gr.Markdown("### 🏷️ Cultural Tag Cloud")
                        cultural_tag_cloud = gr.Textbox(
                            label="Detected Cultural Tags",
                            interactive=False,
                            lines=2
                        )
            
            # Advanced Analysis Accordions
            with gr.Accordion("🔍 Advanced Analysis & Reports", open=False):
                with gr.Row():
                    with gr.Column():
                        archive_index = gr.JSON(
                            label="📑 Cultural Signal Index",
                            elem_classes="card"
                        )
                        
                        disagreement_view = gr.JSON(
                            label="⚠️ Model Disagreement Analysis",
                            elem_classes="card"
                        )
                    
                    with gr.Column():
                        surprise_json = gr.JSON(
                            label="🎭 Cultural Surprise Detection",
                            elem_classes="card"
                        )
                        
                        silence_json = gr.JSON(
                            label="🔇 Cultural Silence Detection",
                            elem_classes="card"
                        )
                
                with gr.Row():
                    trust_json = gr.JSON(
                        label="🛡️ Comprehensive Trust Report",
                        elem_classes="card"
                    )
                    
                    brain_json = gr.JSON(
                        label="🧠 Model Brain Analysis",
                        elem_classes="card"
                    )

            # States
            image_state = gr.State(None)
            context_state = gr.State(None)
            prediction_state = gr.State(None)
            signal_state = gr.State({})

            # Button wiring
            btn.click(
                fn=run_caption,
                inputs=[img, lang, use_esn, compare_models],
                outputs=[
                    out_en, out_local, out_tags, audio, image_state,
                    context_state, comparison_out, reasoning_out,
                    prediction_state, cultural_signal_plot,
                    cultural_tag_cloud, archive_index, signal_state,
                    disagreement_view, surprise_json, silence_json,
                    trust_json, brain_json
                ]
            )

        # ═══════════════════════════════════════════════════════════════
        # Q&A CHATBOT TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("💬 Q&A Chatbot"):
            gr.HTML('<div class="card-header"><h3>🤖 Intelligent Image Q&A</h3></div>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    original_media_qa = gr.Image(
                        label="📸 Current Image",
                        interactive=False,
                        height=280,
                        type="filepath",
                        elem_classes="card"
                    )
                
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="💭 Conversation",
                        height=400,
                        elem_classes="card"
                    )
            
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="💬 Type your question",
                        placeholder="Ask me anything about the image...",
                        lines=1
                    )
                
                with gr.Column(scale=1):
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 Or speak",
                        format="wav"
                    )
            
            with gr.Row():
                qa_btn = gr.Button(
                    "📤 Send Question",
                    variant="primary",
                    elem_classes="primary"
                )
            
            qa_audio = gr.Audio(
                type="filepath",
                label="🔊 Listen to Answer",
                elem_classes="card"
            )

            qa_btn.click(
                fn=process_qa,
                inputs=[text_input, mic_input, chatbot, image_state, context_state],
                outputs=[chatbot, qa_audio, original_media_qa]
            )

        # ═══════════════════════════════════════════════════════════════
        # CULTURAL MEMORY (RAG) TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🏛️ Cultural Memory"):
            gr.HTML('<div class="card-header"><h3>🌟 Cultural Facts & Storytelling</h3></div>')
            
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📚 Accumulated Cultural Knowledge")
                        cultural_facts = gr.Markdown()
                    
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📖 Generated Cultural Story")
                        cultural_story = gr.Textbox(
                            label="Cultural Narrative",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Group(elem_classes="card"):
                        cultural_refs = gr.Textbox(
                            label="🔗 Sources & Knowledge Base",
                            lines=2,
                            interactive=False
                        )

            refresh_rag = gr.Button(
                "🔄 Generate Cultural Memory",
                variant="primary",
                elem_classes="primary"
            )

            refresh_rag.click(
                fn=cultural_rag,
                inputs=[prediction_state, context_state, gr.State("India-wide")],
                outputs=[cultural_facts, cultural_story, cultural_refs]
            )

        # ═══════════════════════════════════════════════════════════════
        # TRUST & SAFETY TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🛡️ Trust & Safety"):
            gr.HTML('<div class="card-header"><h3>📊 Comprehensive Safety Analysis</h3></div>')
            
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🎯 Safety Metrics")
                        bias_score = gr.Slider(
                            0, 1,
                            label="⚠️ Hallucination Risk",
                            interactive=False
                        )
                        vision_score = gr.Slider(
                            0, 1,
                            label="👁️ Vision-Text Consistency",
                            interactive=False
                        )
                        cultural_conf = gr.Slider(
                            0, 1,
                            label="🎭 Cultural Confidence",
                            interactive=False
                        )
                    
                    safety_btn = gr.Button(
                        "🔍 Compute Safety Metrics",
                        variant="primary",
                        elem_classes="primary"
                    )
                
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📋 Detailed Trust Analysis")
                        trust_report_view = gr.JSON(
                            label="Complete Trust Report"
                        )

            safety_btn.click(
                fn=safety_metrics,
                inputs=[prediction_state],
                outputs=[bias_score, vision_score, cultural_conf, trust_report_view]
            )

        # ═══════════════════════════════════════════════════════════════
        # MODEL BRAIN TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🧠 Model Brain"):
            gr.HTML('<div class="card-header"><h3>🔬 Neural Network Interpretability</h3></div>')
            
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🌊 ESN (Echo State Network)")
                        esn_state = gr.Textbox(
                            label="ESN Cultural Activation",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### ⚡ GRU (Gated Recurrent Unit)")
                        gru_state = gr.Textbox(
                            label="GRU Activation",
                            lines=6,
                            interactive=False
                        )
                
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🧩 Cognitive Interpretation")
                        trigger_words = gr.Textbox(
                            label="What Triggered the Models?",
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.Group(elem_classes="card"):
                        brain_report_view = gr.JSON(
                            label="📊 Detailed Brain Analysis"
                        )

            brain_btn = gr.Button(
                "🔍 Analyze Model Cognition",
                variant="primary",
                elem_classes="primary"
            )

            brain_btn.click(
                fn=lambda r, c: model_brain(r, c),
                inputs=[prediction_state, context_state],
                outputs=[esn_state, gru_state, trigger_words, brain_report_view]
            )

        # ═══════════════════════════════════════════════════════════════
        # CULTURAL EVOLUTION TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("⏳ Cultural Evolution"):
            gr.HTML('<div class="card-header"><h3>📈 Temporal Signal Analysis</h3></div>')
            
            with gr.Group(elem_classes="card"):
                gr.Markdown("""
                ### 🌱 Cultural Signal Evolution Over Time
                Track how cultural signals change and develop across your generated captions.
                """)
                
                evolution_plot = gr.Plot(label="Evolution Timeline")
                
                evolution_btn = gr.Button(
                    "🎬 Animate Cultural Evolution",
                    variant="primary",
                    elem_classes="primary"
                )

            evolution_btn.click(
                fn=lambda p: generate_heritage_evolution_snapshot(extract_mode_from_prediction(p)),
                inputs=[prediction_state],
                outputs=evolution_plot
            )

        # ═══════════════════════════════════════════════════════════════
        # REGIONAL CULTURE TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🌍 Regional Culture"):
            gr.HTML('<div class="card-header"><h3>🗺️ Cultural Adaptation Engine</h3></div>')
            
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📝 Original Caption")
                        original_reg = gr.Textbox(
                            label="As Generated",
                            lines=4,
                            interactive=False,
                            value="Generate a caption first →"
                        )
                
                with gr.Column():
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🎨 Culturally Adapted")
                        regional_caption = gr.Textbox(
                            label="Adapted Version",
                            lines=6,
                            interactive=False
                        )
            
            with gr.Row():
                with gr.Column(scale=2):
                    region = gr.Dropdown(
                        choices=[
                            "Japan", "Italy", "Mexico", "France", "Thailand", "Morocco",
                            "Brazil", "Korea", "Turkey", "Lebanon", "Ethiopia", "Vietnam",
                            "Peru", "Greece", "Spain", "Nigeria", "Indonesia", "Sweden",
                            "United States (Southern)", "United States (New York style)",
                            "Generic international fusion"
                        ],
                        label="🌏 Select Target Culture",
                        value="Japan"
                    )
                
                with gr.Column(scale=1):
                    regional_btn = gr.Button(
                        "🔄 Adapt to Culture",
                        variant="primary",
                        elem_classes="primary"
                    )

            regional_btn.click(
                fn=lambda cap, cult: (cap, regional_adapt(cap, cult)),
                inputs=[context_state, region],
                outputs=[original_reg, regional_caption]
            )

        # ═══════════════════════════════════════════════════════════════
        # ACCESSIBILITY TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("♿ Accessibility"):
            gr.HTML('<div class="card-header"><h3>👁️ Screen Reader Optimization</h3></div>')
            
            with gr.Group(elem_classes="card"):
                gr.Markdown("""
                ### 📖 Enhanced Accessibility Description
                Optimized for screen readers and assistive technologies.
                """)
                
                blind_caption = gr.Textbox(
                    label="Detailed Description",
                    lines=7,
                    interactive=False
                )
                
                blind_audio = gr.Audio(
                    label="🔊 Audio Narration",
                    type="filepath"
                )

            accessibility_btn = gr.Button(
                "♿ Generate Accessibility Mode",
                variant="primary",
                elem_classes="primary"
            )

            accessibility_btn.click(
                fn=accessibility_description,
                inputs=context_state,
                outputs=[blind_caption, blind_audio]
            )

        # ═══════════════════════════════════════════════════════════════
        # HERITAGE MAP TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🗺️ Heritage Map"):
            gr.HTML('<div class="card-header"><h3>🌐 Cultural Heritage Network</h3></div>')
            
            with gr.Group(elem_classes="card"):
                gr.Markdown("""
                ### 🕸️ Signal Connection Graph
                Visualize how cultural signals relate to each other across your generated content.
                """)
                
                heritage_graph = gr.Plot(
                    label="Heritage Network Visualization"
                )
                
                heritage_out = gr.JSON(
                    label="📊 Network Statistics"
                )

            map_btn = gr.Button(
                "🗺️ Generate Heritage Map",
                variant="primary",
                elem_classes="primary"
            )

            map_btn.click(
                fn=lambda _, p: generate_heritage_graph(_, extract_mode_from_prediction(p)),
                inputs=[context_state, prediction_state],
                outputs=[heritage_graph, heritage_out]
            )

        # ═══════════════════════════════════════════════════════════════
        # WHAT-IF CULTURAL FRAME TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🔮 What-If Analysis"):
            gr.HTML('<div class="card-header"><h3>🎭 Counterfactual Cultural Frames</h3></div>')
            
            with gr.Group(elem_classes="card"):
                gr.Markdown("""
                ### 🌀 Re-interpret Through Different Lenses
                See how cultural signals would shift in alternative contexts.
                """)
                
                frame = gr.Dropdown(
                    choices=list(COUNTERFACTUAL_FRAMES.keys()),
                    label="🎬 Select Cultural Frame",
                    value=list(COUNTERFACTUAL_FRAMES.keys())[0] if COUNTERFACTUAL_FRAMES else None
                )
                
                cf_out = gr.JSON(
                    label="🔄 Counterfactual Interpretation"
                )

            frame.change(
                fn=lambda s, f: counterfactual_from_vector(s, f),
                inputs=[signal_state, frame],
                outputs=cf_out
            )

        # ═══════════════════════════════════════════════════════════════
        # SIMILAR IMAGES TAB
        # ═══════════════════════════════════════════════════════════════
        with gr.Tab("🖼️ Similar Images"):
            gr.HTML('<div class="card-header"><h3>🔍 Visual Discovery</h3></div>')
            
            with gr.Group(elem_classes="card"):
                gr.Markdown("""
                ### 🌐 Find Similar Images Online
                Powered by Pexels API - Discover visually similar content.
                """)
                
                similar_gallery = gr.Gallery(
                    label="Similar Images",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                    elem_classes="card"
                )
                
                refresh_btn = gr.Button(
                    "🔄 Refresh Similar Images",
                    variant="primary",
                    elem_classes="primary"
                )

            refresh_btn.click(
                fn=search_similar_images,
                inputs=context_state,
                outputs=similar_gallery
            )

    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: #7F8C8D; font-size: 0.9rem;">
        <p>Cultural AI Explorer • v2.0</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            Powered by ESN/GRU • Ollama • SQLite • Pexels API
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)