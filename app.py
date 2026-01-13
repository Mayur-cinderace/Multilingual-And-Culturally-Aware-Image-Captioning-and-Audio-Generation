# app.py – Cultural AI Explorer with SQLite DB + Ollama (local)
import gradio as gr
from gtts import gTTS
from pathlib import Path
import speech_recognition as sr
import os
import sqlite3
import json
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
import datetime

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
    c.execute('''
        CREATE TABLE IF NOT EXISTS cultural_cues (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            mode                     TEXT NOT NULL,
            confidence               REAL DEFAULT 0.0,
            english_caption          TEXT,
            local_caption            TEXT,
            hashtags                 TEXT,
            detected_dish            TEXT,
            detected_festival        TEXT,
            detected_region_hint     TEXT,
            short_description        TEXT,
            added_at                 DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_hash               TEXT
        )
    ''')
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

from cpu_cultural_enrich import generate_base_caption, translate_to_language
from cultural_context import CulturalContextManager
from hashtag_generator import generate_hashtags
from translation_refiner import naturalize_translation

training_samples = [
    ("A bowl of spicy curry garnished with fresh herbs.", "food_traditional"),
    ("Aromatic masala dish served in a traditional bowl.", "food_traditional"),
    ("Gravy with spices and vegetables.", "food_traditional"),
    ("Spiced rice garnished with nuts.", "food_traditional"),
    ("Traditional meal with rich flavors.", "food_traditional"),
    ("People celebrating with lights and sweets during festival.", "festival_context"),
    ("Ritual offerings in a celebration.", "festival_context"),
    ("Festival gathering with colorful decorations.", "festival_context"),
    ("Traditional dance during celebratory event.", "festival_context"),
    ("Ceremony with bright colors and music.", "festival_context"),
    ("Family having a simple home meal.", "daily_life"),
    ("Daily routine with breakfast in bowl.", "daily_life"),
    ("Home-cooked dish served to family.", "daily_life"),
    ("Everyday gathering around the table.", "daily_life"),
    ("Routine meal in a casual setting.", "daily_life"),
    ("A landscape with mountains and sky.", "generic"),
    ("Abstract art piece on wall.", "generic"),
    ("City street with cars.", "generic"),
    ("Book on a table.", "generic"),
    ("Random object in room.", "generic")
]

cultural_manager = CulturalContextManager()
cultural_manager.train(training_samples)

CULTURAL_CONFIDENCE_THRESHOLD = 0.55

CULTURAL_TEMPLATES = {
    "food_traditional": "This reflects traditional cooking styles, often featuring rich spices and shared meals.",
    "festival_context": "Such scenes are common in cultural festivals and communal celebrations.",
    "daily_life": "This is typically part of everyday routines and family interactions.",
    "generic": ""
}


# ------------------------------------------------------------
# Entity Extraction Helper
# ------------------------------------------------------------
def light_cultural_extraction(caption: str, mode: str, confidence: float) -> dict:
    if mode in ["food_traditional", "festival_context"]:
        pass
    elif confidence < 0.20:
        return {"dish": "", "festival": "", "region": "", "desc": ""}
    system = """
You are a precise cultural entity extractor for Indian contexts.
Only return valid JSON object. Do NOT explain. Do NOT add extra text.
Return empty strings if nothing is clearly indicated.

Example output:
{"dish": "Butter Chicken", "festival": "Diwali", "region": "Punjab", "desc": "Rich tomato-based curry with cream and butter"}
"""

    prompt = f"""
Caption: {caption}
Mode: {mode} (confidence {confidence:.2f})

Extract exactly these keys:
- dish: most likely dish/food name
- festival: most likely festival (if any)
- region: most likely region/state
- desc: one short descriptive phrase (max 12 words)
"""

    try:
        raw = ollama_generate(prompt, system=system, temperature=0.0, max_tokens=180)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)
        return {
            "dish": data.get("dish", ""),
            "festival": data.get("festival", ""),
            "region": data.get("region", ""),
            "desc": data.get("desc", "")
        }
    except Exception:
        return {"dish": "", "festival": "", "region": "", "desc": ""}


# ------------------------------------------------------------
# DB Helper Functions
# ------------------------------------------------------------
def get_cultural_facts(mode: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Using new schema columns
    c.execute("""
        SELECT detected_dish, short_description, detected_region_hint, detected_festival 
        FROM cultural_cues 
        WHERE mode = ? AND detected_dish != ''
        ORDER BY added_at DESC LIMIT 12
    """, (mode,))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return "No cultural entries found for this mode yet."

    facts = "\n".join([f"• {row[0]}: {row[1]} ({row[2]})" for row in rows if row[0]])
    return f"**Recent Cultural Entries**\n{facts}"


def add_cultural_entry(mode, name, description, region, festival, story, recipe_summary):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO cultural_cues (
                mode, detected_dish, short_description, detected_region_hint, detected_festival
            ) VALUES (?, ?, ?, ?, ?)
        ''', (mode, name, description, region, festival))
        conn.commit()
        conn.close()
        return "Entry added successfully!"
    except Exception as e:
        return f"Error adding entry: {str(e)}"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def visually_ambiguous_food(caption: str) -> bool:
    c = caption.lower()
    return any(w in c for w in ["round", "smooth", "plain", "boiled"]) and \
           any(w in c for w in ["plate", "bowl", "served"])


def search_similar_images(caption: str, num_results=8):
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_API_KEY:
        return ["https://via.placeholder.com/300x200?text=Add+PEXELS_API_KEY+in+.env"]

    query = caption.strip().replace(" ", "+")
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={num_results}"
    headers = {"Authorization": PEXELS_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        data = response.json()
        photos = data.get("photos", [])
        urls = [p["src"]["medium"] for p in photos if "src" in p and "medium" in p["src"]]
        return urls or ["https://via.placeholder.com/300x200?text=No+Images+Found"]
    except Exception as e:
        print(f"[Pexels] {e}")
        return ["https://via.placeholder.com/300x200?text=Error+Loading+Images"]


# ------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------
def run_caption(image, language, use_esn, compare_models=False):
    if image is None:
        return "", "", "", None, None, None, "", "", None  # 9 outputs now

    image_path = "temp.jpg"
    image.save(image_path)

    base_caption = generate_base_caption(image_path).strip()

    # ── Cultural classification ────────────────────────────────
    cultural_line = ""
    comparison_text = ""
    reasoning_trace = ""
    mode = "generic"
    confidence = 0.0
    results = {  # default fallback
        "esn": {"mode": "generic", "confidence": 0.0},
        "gru": {"mode": "generic", "confidence": 0.0},
        "combined_mode": "generic"
    }

    if use_esn:
        results = cultural_manager.predict(base_caption)
        
        if compare_models:
            comparison_text = (
                f"ESN: {results['esn']['mode']} ({results['esn']['confidence']:.2f}) | "
                f"GRU: {results['gru']['mode']} ({results['gru']['confidence']:.2f})"
            )
        else:
            comparison_text = ""

        mode = results["esn"]["mode"]
        confidence = results["esn"]["confidence"]

        if confidence >= CULTURAL_CONFIDENCE_THRESHOLD:
            cultural_line = CULTURAL_TEMPLATES.get(mode, "")
            if visually_ambiguous_food(base_caption):
                cultural_line = "Items like this are often seen in traditional sweets or dishes, especially in cultural contexts."

    final_caption_en = (base_caption.rstrip(".") + ". " + cultural_line).strip() if cultural_line else base_caption

    # ── Translate ───────────────────────────────────────────────
    translated = translate_to_language(final_caption_en, language)
    local_caption = naturalize_translation(translated, language) if language != "English" else translated

    # ── Hashtags ────────────────────────────────────────────────
    hashtags_str = generate_hashtags(final_caption_en, min_tags=2)

    # ── Cultural entity extraction ──────────────────────────────
    entities = light_cultural_extraction(final_caption_en, mode, confidence)

    # ── Auto-save to DB ─────────────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO cultural_cues (
                mode, confidence, english_caption, local_caption, hashtags,
                detected_dish, detected_festival, detected_region_hint, short_description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            mode,
            round(confidence, 3),
            final_caption_en,
            local_caption,
            hashtags_str,
            entities["dish"],
            entities["festival"],
            entities["region"],
            entities["desc"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB auto-save failed: {str(e)}")

    # ── Audio Narration ─────────────────────────────────────────
    audio_path = OUT_AUDIO / "caption.mp3"
    audio_file = None
    try:
        tts_text = local_caption if local_caption.strip() else final_caption_en
        gTTS(tts_text, lang=TTS_LANG.get(language, "en")).save(audio_path)
        audio_file = str(audio_path)
    except Exception as e:
        print(f"TTS failed: {e}")

    # ── Return all 9 values ─────────────────────────────────────
    return (
        final_caption_en,                           # out_en
        local_caption,                              # out_local
        hashtags_str,                               # out_tags
        audio_file,                                 # audio
        str(image_path),                            # image_state
        final_caption_en + (" " + cultural_line if cultural_line else ""),  # context_state
        comparison_text,                            # comparison_out
        reasoning_trace,                            # reasoning_out
        results                                     # prediction_state (the dict!)
    )


def process_qa(question, audio_input, history, image_path, context):
    if audio_input is not None:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_input) as source:
            audio_data = recognizer.record(source)
            try:
                question = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                question = "Sorry, could not understand audio."
            except sr.RequestError:
                question = "Sorry, speech service unavailable."

    if not question or not image_path:
        return history + [["", "Please generate a caption first and ask a question about the image."]], None, None

    context_trim = context[:800]
    system_prompt = f"You are a helpful cultural assistant. The image shows: {context_trim}. Answer in a detailed, friendly way."

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


def cultural_rag(mode, caption, region="India-wide"):
    facts_md = get_cultural_facts(mode)
    prompt = f"""
You are a cultural storyteller.
Context:
Caption: {caption}
Region preference: {region}
Known facts:
{facts_md}

Write an engaging 3–5 sentence cultural story or explanation connecting the caption to Indian heritage.
Be accurate, respectful, and vivid.
"""
    try:
        story = ollama_generate(prompt, temperature=0.65, max_tokens=450)
        return facts_md, story, "SQLite DB + Ollama"
    except Exception as e:
        return facts_md, f"Could not generate story: {str(e)}", "N/A"


def regional_adapt(caption, target_culture):
    if not caption or not target_culture:
        return "Please generate a caption first and select a culture."

    prompt = f"""
Rewrite the following image caption in a culturally authentic style as if it were described by someone from **{target_culture}**.
Use typical language patterns, expressions, ingredients, presentation styles or cultural references common in {target_culture}.
Keep the core content the same, but adapt the tone, wording and details to feel native to that culture.

Original caption:
{caption}

Rewritten version:
"""

    try:
        adapted = ollama_generate(prompt, temperature=0.75, max_tokens=350)
        return adapted.strip()
    except Exception as e:
        return f"[Error adapting caption] {str(e)}"

def accessibility_description(caption):
    prompt = f"Describe this image in very clear, simple, structured language suitable for a screen reader or visually impaired user:\n\n{caption}"
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


def generate_heritage_graph(context, mode):
    G = nx.Graph()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT detected_dish, detected_festival, detected_region_hint
        FROM cultural_cues
        WHERE mode = ?
        ORDER BY added_at DESC
        LIMIT 30
    """, (mode,))
    rows = c.fetchall()
    print(f"[DEBUG Heritage] Mode: {mode}")
    print(f"[DEBUG Heritage] Rows found: {len(rows)}")
    if rows:
        print("[DEBUG Heritage] First few rows:", rows[:3])
    else:
        print("[DEBUG Heritage] No matching entries — check if detected_dish is filled")
    conn.close()

    print(f"[Heritage Debug] Mode={mode}, Rows={len(rows)}")   # ← keep for now

    added_dishes = set()

    for dish, festival, region in rows:
        dish = (dish or "").strip()
        if not dish:
            continue

        if dish in added_dishes:
            continue
        added_dishes.add(dish)

        G.add_node(dish, type="dish")

        festival = (festival or "").strip()
        if festival:
            G.add_node(festival, type="festival")
            G.add_edge(dish, festival)

        region = (region or "").strip()
        if region:
            G.add_node(region, type="region")
            G.add_edge(dish, region)

    if len(G.nodes) == 0:
        # Fallback: show a message node
        G.add_node("No cultural entries yet\nTry processing some food images", type="info")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(G, k=0.9, iterations=40)

    # Color by type
    node_colors = []
    for n, d in G.nodes(data=True):
        t = d.get("type", "dish")
        if t == "dish":    node_colors.append("#a8dadc")
        elif t == "festival": node_colors.append("#e63946")
        elif t == "region":   node_colors.append("#457b9d")
        else:              node_colors.append("#cccccc")

    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=2200,
        font_size=9,
        font_weight="bold",
        edge_color="#999999",
        width=1.2,
        alpha=0.9
    )

    return fig, [{"dish": d, "festival": f, "region": r} for d, f, r in rows]



    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=1800, font_size=10, ax=ax)
    return fig, [{"name": n, "festival": f} for n, f in rows]


def safety_metrics(caption, mode, conf):
    hallucination_risk = 0.1 if conf > 0.8 else 0.5
    vision_align = 0.9 if any(w in caption.lower() for w in ["food", "dish", "meal", "festival", "thali"]) else 0.6
    return hallucination_risk, vision_align, conf


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
        # Caption Generation tab ........................................
        with gr.Tab("Caption Generation"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=360):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📤 Upload & Settings")
                        img = gr.Image(type="pil", label="Upload Image", height=360)
                        lang = gr.Dropdown(choices=list(TTS_LANG.keys()), value="English", label="Output Language")
                        use_esn = gr.Checkbox(value=True, label="Inject Cultural Context (ESN)")
                        compare_models = gr.Checkbox(value=False, label="Compare ESN vs GRU")
                        btn = gr.Button("✨ Generate Caption", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=520):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📝 Generated Output")
                        out_en = gr.Textbox(label="Final Caption (English)", lines=3, interactive=False, show_copy_button=True)
                        out_local = gr.Textbox(label="Translated Caption", lines=3, interactive=False, show_copy_button=True)
                        out_tags = gr.Textbox(label="Suggested Hashtags", lines=2, interactive=False, show_copy_button=True)
                        comparison_out = gr.Textbox(label="Model Comparison (ESN vs GRU)", lines=2, interactive=False)
                        reasoning_out = gr.Textbox(label="Reasoning Trace", lines=3)
                        gr.Markdown("### 🔊 Audio Narration")
                        audio = gr.Audio(type="filepath", label="Listen to Caption")

            image_state = gr.State(None)
            context_state = gr.State(None)
            prediction_state = gr.State(None)   # will hold the full dict from predict()

            btn.click(
                fn=run_caption,
                inputs=[img, lang, use_esn, compare_models],
                outputs=[out_en, out_local, out_tags, audio, image_state, context_state, comparison_out, reasoning_out, prediction_state]
            )

        with gr.Tab("Q&A Chatbot"):
            with gr.Row():
                with gr.Column():
                    original_media_qa = gr.Image(label="Uploaded Media", interactive=False, height=200, type="filepath")
                    chatbot = gr.Chatbot(label="Chat with the Image")
                    text_input = gr.Textbox(label="Type your question")
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Or speak your question")
                    qa_btn = gr.Button("Send")

                    qa_audio = gr.Audio(type="filepath", label="Listen to Answer")

                    qa_btn.click(
                        fn=process_qa,
                        inputs=[text_input, mic_input, chatbot, image_state, context_state],
                        outputs=[chatbot, qa_audio, original_media_qa]
                    )

        with gr.Tab("Cultural Memory (RAG)"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🌟 Cultural Facts & Story")
                    cultural_facts = gr.Markdown()
                    cultural_story = gr.Textbox(label="Cultural Story", lines=5)
                    cultural_refs = gr.Textbox(label="Sources / Knowledge", lines=3)

            refresh_rag = gr.Button("Generate Cultural Memory")

            refresh_rag.click(
                fn=cultural_rag,
                inputs=[gr.State("food_traditional"), context_state, gr.State("India-wide")],
                outputs=[cultural_facts, cultural_story, cultural_refs]
            )

            with gr.Group():
                gr.Markdown("### ➕ Add New Cultural Entry to DB (manual)")
                new_mode = gr.Dropdown(["food_traditional", "festival_context", "daily_life", "generic"], label="Mode")
                new_name = gr.Textbox(label="Name (e.g. Biryani)")
                new_desc = gr.Textbox(label="Description")
                new_region = gr.Textbox(label="Region")
                new_festival = gr.Textbox(label="Festival")
                new_story = gr.Textbox(label="Story", lines=3)
                new_recipe = gr.Textbox(label="Recipe Summary", lines=3)
                add_btn = gr.Button("Add to Database")

                add_status = gr.Textbox(label="Status")

            add_btn.click(
                fn=add_cultural_entry,
                inputs=[new_mode, new_name, new_desc, new_region, new_festival, new_story, new_recipe],
                outputs=add_status
            )

        with gr.Tab("Trust & Safety"):
            with gr.Row():
                bias_score = gr.Slider(0,1,label="Cultural Hallucination Risk")
                vision_score = gr.Slider(0,1,label="Vision-Text Consistency")
                cultural_conf = gr.Slider(0,1,label="Cultural Confidence")

            safety_btn = gr.Button("Compute Safety Metrics")

            safety_btn.click(
                fn=safety_metrics,
                inputs=[context_state, gr.State("food_traditional"), gr.State(0.8)],
                outputs=[bias_score, vision_score, cultural_conf]
            )

        with gr.Tab("Model Brain"):
            with gr.Row():
                esn_state = gr.Textbox(label="ESN Cultural Activation", lines=5)
                gru_state = gr.Textbox(label="GRU Activation", lines=5)
                trigger_words = gr.Textbox(label="Cultural Triggers", lines=3)

            brain_btn = gr.Button("Show Model Brain")

            brain_btn.click(
                fn=model_brain,
                inputs=[prediction_state],
                outputs=[esn_state, gru_state, trigger_words]
            )

        # Inside the "Regional Culture" tab block
        with gr.Tab("Regional Culture"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Caption")
                    original_reg = gr.Textbox(label="Original (as generated)", lines=4, interactive=False, value="Generate a caption first →")
                
                with gr.Column():
                    gr.Markdown("### Adapted to Selected Culture")
                    regional_caption = gr.Textbox(label="Adapted version", lines=6, interactive=False)

            with gr.Row():
                region = gr.Dropdown(
                    choices=[
                    "Japan", "Italy", "Mexico", "France", "Thailand", "Morocco",
                    "Brazil", "Korea", "Turkey", "Lebanon", "Ethiopia", "Vietnam",
                    "Peru", "Greece", "Spain", "Nigeria", "Indonesia", "Sweden",
                    "United States (Southern)", "United States (New York style)",
                    "Generic international fusion"],
                    label="Select Target Culture / Country",
                    value="Japan"
                )
                regional_btn = gr.Button("Adapt to This Culture", variant="primary")

            regional_btn.click(
                fn=lambda cap, cult: (cap, regional_adapt(cap, cult)),
                inputs=[context_state, region],
                outputs=[original_reg, regional_caption]
            )

        with gr.Tab("Accessibility"):
            with gr.Row():
                blind_caption = gr.Textbox(label="Screen Reader Friendly Description", lines=5)
                blind_audio = gr.Audio(label="Audio Narration for Blind Users")

            accessibility_btn = gr.Button("Generate Accessibility Mode")

            accessibility_btn.click(
                fn=accessibility_description,
                inputs=context_state,
                outputs=[blind_caption, blind_audio]
            )

        with gr.Tab("Cultural Heritage Map"):
            with gr.Row():
                heritage_graph = gr.Plot(label="Heritage Graph Visualization")
                heritage_out = gr.JSON(label="Heritage Data")

            map_btn = gr.Button("Generate Heritage Map")

            map_btn.click(
                fn=generate_heritage_graph,
                inputs=[context_state, gr.State("food_traditional")],
                outputs=[heritage_graph, heritage_out]
            )

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

# Important: share=True creates public link (good for testing / demo)
# For local-only → keep share=False or remove the argument
demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)