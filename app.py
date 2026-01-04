# app.py (Updated – Q&A now uses Groq API only, no local model, no toggle)
import gradio as gr
from gtts import gTTS
from pathlib import Path
import speech_recognition as sr
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # Loads GROQ_API_KEY from .env

from cpu_cultural_enrich import generate_base_caption, translate_to_language
from esn_cultural_context import CulturalContextController
from hashtag_generator import generate_hashtags
from translation_refiner import naturalize_translation

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
OUT_AUDIO = Path("captions_output_cpu/audio")
OUT_AUDIO.mkdir(parents=True, exist_ok=True)

TTS_LANG = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa",
    "Urdu": "ur", "Odia": "or", "Assamese": "as", "Nepali": "ne", "Konkani": "gom", "Sanskrit": "sa"
}

# ESN setup (unchanged)
esn = CulturalContextController()
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
esn.train(training_samples)

CULTURAL_CONFIDENCE_THRESHOLD = 0.55

CULTURAL_TEMPLATES = {
    "food_traditional": "This reflects traditional cooking styles, often featuring rich spices and shared meals.",
    "festival_context": "Such scenes are common in cultural festivals and communal celebrations.",
    "daily_life": "This is typically part of everyday routines and family interactions.",
    "generic": ""
}

# Groq setup (only mode for Q&A)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

groq_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # Fast & good; can change to "mixtral-8x7b-32768" etc.
    temperature=0.7,
    max_tokens=400
)

# ------------------------------------------------------------
# Visual ambiguity detector (unchanged)
# ------------------------------------------------------------
def visually_ambiguous_food(caption: str) -> bool:
    c = caption.lower()
    return any(w in c for w in ["round", "smooth", "plain", "boiled"]) and any(w in c for w in ["plate", "bowl", "served"])

# ------------------------------------------------------------
# Caption Generation (unchanged)
# ------------------------------------------------------------
def run_caption(image, language, use_esn):
    if image is None:
        return "", "", "", None, None, None

    image_path = "temp.jpg"
    image.save(image_path)

    base_caption = generate_base_caption(image_path).strip()

    cultural_line = ""
    if use_esn:
        mode, confidence = esn.predict_mode(base_caption)
        if confidence >= CULTURAL_CONFIDENCE_THRESHOLD:
            if visually_ambiguous_food(base_caption):
                cultural_line = "Items like this are often seen in traditional sweets or dishes, especially in cultural contexts."
            else:
                cultural_line = CULTURAL_TEMPLATES.get(mode, "")

    final_caption_en = base_caption.rstrip(".") + ". " + cultural_line if cultural_line else base_caption

    translated = translate_to_language(final_caption_en, language)
    local_caption = naturalize_translation(translated, language) if language != "English" else translated

    hashtags = generate_hashtags(final_caption_en, min_tags=2)

    audio_path = OUT_AUDIO / "caption.mp3"
    audio_file = None
    try:
        gTTS(local_caption, lang=TTS_LANG.get(language, "en")).save(audio_path)
        audio_file = str(audio_path)
    except Exception:
        pass

    return final_caption_en, local_caption, hashtags, audio_file, image_path, final_caption_en + " " + cultural_line

# ------------------------------------------------------------
# Q&A Processing (Groq only)
# ------------------------------------------------------------
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

    if not question or image_path is None:
        return history + [["", "Please generate a caption first and ask a question about the image."]], None

    # System prompt with image context
    system_prompt = f"You are a helpful cultural assistant. The image shows: {context}. Answer in a detailed, friendly way."

    try:
        prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
        response = groq_llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        print(f"[ERROR] Groq API failed: {e}")
        answer = "Sorry, there was an issue processing your question. Please try again."

    answer = answer.capitalize()
    if not answer.endswith(('.', '!', '?')):
        answer += '.'

    audio_path = OUT_AUDIO / "answer.mp3"
    audio_file = None
    try:
        gTTS(answer, lang="en").save(audio_path)
        audio_file = str(audio_path)
    except Exception:
        pass

    history.append([question, answer])
    return history, audio_file

# ------------------------------------------------------------
# UI (Q&A tab simplified – no Groq toggle)
# ------------------------------------------------------------
with gr.Blocks(
    title="Cultural Image Captioner",
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
        <h1>🖼️ Cultural Image Captioner</h1>
        <p>Low-resource • ESN-guided • Multilingual • Culturally aware • With Q&A Chatbot</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Caption Generation"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=360):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📤 Upload & Settings")
                        img = gr.Image(type="pil", label="Upload Image", height=360)
                        lang = gr.Dropdown(choices=list(TTS_LANG.keys()), value="English", label="Output Language")
                        use_esn = gr.Checkbox(value=True, label="Inject Cultural Context (ESN)")
                        btn = gr.Button("✨ Generate Caption", variant="primary", size="lg")

                with gr.Column(scale=2, min_width=520):
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 📝 Generated Output")
                        out_en = gr.Textbox(label="Final Caption (English)", lines=3, interactive=False, show_copy_button=True)
                        out_local = gr.Textbox(label="Translated Caption", lines=3, interactive=False, show_copy_button=True)
                        out_tags = gr.Textbox(label="Suggested Hashtags", lines=2, interactive=False, show_copy_button=True)
                        gr.Markdown("### 🔊 Audio Narration")
                        audio = gr.Audio(type="filepath", label="Listen to Caption")

            image_state = gr.State(None)
            context_state = gr.State(None)

            btn.click(
                fn=run_caption,
                inputs=[img, lang, use_esn],
                outputs=[out_en, out_local, out_tags, audio, image_state, context_state]
            )

        with gr.Tab("Q&A Chatbot"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(label="Chat with the Image")
                    text_input = gr.Textbox(label="Type your question")
                    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Or speak your question")
                    qa_btn = gr.Button("Send")

                    qa_audio = gr.Audio(type="filepath", label="Listen to Answer")

                    qa_btn.click(
                        fn=process_qa,
                        inputs=[text_input, mic_input, chatbot, image_state, context_state],
                        outputs=[chatbot, qa_audio]
                    )

demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)