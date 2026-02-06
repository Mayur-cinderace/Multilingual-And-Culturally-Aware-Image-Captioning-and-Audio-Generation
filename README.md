# 🌍 Multilingual & Culturally-Aware Image Captioning & Audio Generation

A comprehensive AI system that generates **culturally-grounded, multilingual image captions** with advanced cultural context modeling, trust & safety analysis, and audio generation capabilities.

## ✨ Key Features

### 🖼️ Intelligent Image Captioning
- **Florence-2 Vision Model**: High-quality detailed image descriptions
- **Cultural Context Classification**: Determines cultural mode (Traditional, Festival, Daily Life, Generic)
- **ESN & GRU Models**: Echo State Networks and Gated Recurrent Units for cultural signal analysis
- **Multilingual Output**: Supports 20+ languages with automatic translation

### 🌏 Cultural Adaptation Engine
- **Regional Culture Variants**: Automatically adapts captions to 20+ cultural contexts (Japan, Italy, Mexico, Thailand, India, etc.)
- **Cultural Injection**: Enhances captions with culturally-relevant context and nuance
- **Cultural Signal Analysis**: Detects and quantifies cultural markers in images
- **Heritage-Aware Captioning**: Recognizes traditional practices, rituals, and cultural elements

### 🎯 Advanced Analysis Modules
- **Trust & Safety**: Multi-dimensional trust assessment of AI outputs
- **Cultural Surprise Detection**: Identifies violations of learned cultural expectations
- **Cultural Silence Detection**: Meaningful absence of cultural markers
- **Disagreement Analysis**: Compares ESN vs GRU model predictions
- **Signal Strength Analysis**: Comprehensive cultural signal interpretation

### 🔊 Audio & Multimodal Features
- **Text-to-Speech (gTTS)**: Convert generated captions to natural speech
- **Speech Recognition**: Input via voice with audio transcription
- **3D Visualizations**: ESN reservoir states, GRU evolution, and model comparisons
- **Cultural Context RAG**: Retrieval-augmented generation grounded in cultural facts

### 💾 Persistent Storage
- **SQLite Database**: Caches captions, cultural facts, and historical data
- **JSON Output**: Structured results for integration with other systems
- **Cultural Library**: Indexable database of heritage, rituals, and traditions

### 🎨 Interactive Web Interface
- **Gradio-based UI**: User-friendly interface with multiple analysis tabs
- **Real-time Processing**: Live caption generation and cultural adaptation
- **Visual Analytics**: Signal plots, tag clouds, and confusion matrices
- **Model Comparison**: Side-by-side ESN vs GRU performance analysis

## 🏗️ Project Structure

```
ANNDL/
├── app.py                              # Main Gradio application interface
├── 
├── Core Modules
├── ├── cultural_context.py             # ESN & GRU cultural controllers
├── ├── cultural_analysis_enhanced.py   # Cultural signal & surprise detection
├── ├── cultural_rag.py                 # Retrieval-augmented generation for cultural facts
├── ├── trust_safety_enhanced.py        # Trust & safety assessment framework
├── ├── cpu_cultural_enrich.py          # Florence-2 captioning + translation
├── │
├── Feature Extraction & Generation
├── ├── gen_extraction.py               # Genre/context feature extraction
├── ├── idd_extraction.py               # Inter-domain distribution analysis
├── ├── ifd_extraction.py               # Inter-feature distribution analysis
├── ├── people_extraction.py            # People detection and analysis
├── ├── hashtag_generator.py            # Auto-generates cultural hashtags
├── ├── lazy_refiner.py                 # Post-processing refinement
├── ├── translation_refiner.py          # Improves translation quality
├── │
├── Neural Network Models
├── ├── esn_gru_3d.py                   # 3D ESN-GRU architecture
├── ├── esn_gru_visualization.py        # 3D visualizations & animations
├── ├── esn_plot.py                     # ESN performance plots
├── │
├── Evaluation & Benchmarking
├── ├── evaluate.py                     # General evaluation framework
├── ├── evaluate_cultural_models.py     # Cultural model evaluation metrics
├── ├── evaluate_caption_metrics.py     # BLEU, METEOR, CIDEr metrics
├── │
├── Utilities & Support
├── ├── ram_profiler.py                 # Memory usage profiling
├── ├── ollama_proxy.py                 # Ollama LLM service wrapper
├── ├── org.py                          # Organization utilities
├── ├── epoch.py                        # Training utilities
├── ├── training_data.py                # Data loading & preprocessing
├── ├── run_model.py                    # Single batch processing
├── ├── run_visualizations.py           # Generate all visualizations
├── ├── run_cognition_viz.py            # Cognition-focused visualizations
├── │
├── Data & Configuration
├── ├── metadata.json                   # Image metadata
├── ├── metadata.csv                    # Structured metadata
├── ├── references.json                 # Reference captions
├── ├── baseline.json                   # Baseline results
├── ├── cultural_data.db                # SQLite cultural database
├── ├── requirements.txt                # Python dependencies
├── │
├── Output & Results
├── ├── captions_output_cpu/            # Generated captions (JSON)
├── ├── evaluation_outputs/             # Evaluation metrics & plots
├── ├── caption_results.json            # Aggregated caption results
├── ├── caption_metrics_results.csv     # Metric evaluations
├── ├── model_output.json               # Model predictions
├── │
├── Supporting Resources
├── ├── eval/                           # Test images for evaluation
├── ├── training_logs/                  # Training history & logs
├── ├── viz_server/                     # Visualization server assets
├── ├── curate/                         # Curated datasets
├── ├── curate_categorized/             # Categorized curated data
├── ├── pycocoevalcap/                  # COCO evaluation toolkit
├── └── hf_cache/                       # HuggingFace model cache
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Ollama (for local LLM integration)
- 8GB+ RAM recommended

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Mayur-cinderace/Multilingual-And-Culturally-Aware-Image-Captioning-and-Audio-Generation.git
cd ANNDL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.2:3b

# HuggingFace Cache
HF_HOME=./hf_cache
TRANSFORMERS_CACHE=./hf_cache/transformers
HF_HUB_CACHE=./hf_cache/hub

# Database
DB_PATH=cultural_data.db
```

### 3. Start Ollama (Optional but Recommended)

```bash
# Install Ollama from https://ollama.ai
# Then run:
ollama serve
# In another terminal:
ollama pull llama3.2:3b
```

### 4. Run the Application

```bash
# Start the main Gradio interface
python app.py

# The application will be available at http://localhost:7860
```

## 💻 Usage Examples

### Basic Image Captioning

```python
from cpu_cultural_enrich import generate_base_caption
from cultural_context import CulturalContextManager

# Initialize manager
cultural_manager = CulturalContextManager()

# Generate caption
caption = generate_base_caption("image.jpg")
print(f"Caption: {caption}")

# Predict cultural mode
prediction = cultural_manager.predict(caption)
print(f"Mode: {prediction['combined_mode']}")
```

### Cultural Adaptation

```python
from app import regional_adapt

# Adapt caption to specific culture
adapted = regional_adapt("A traditional thali ceremony", "Japan")
print(f"Adapted: {adapted}")
```

### Trust & Safety Analysis

```python
from trust_safety_enhanced import CulturalTrustAnalyzer

analyzer = CulturalTrustAnalyzer()
caption = "Traditional wedding feast with elaborate thali"
trust_report = analyzer.comprehensive_assessment(
    caption, {"mode": "Traditional", "confidence": 0.92}
)
print(f"Trust Score: {trust_report['overall_trust_score']}")
```

### Model Evaluation

```bash
# Evaluate cultural models
python evaluate_cultural_models.py

# Evaluate caption metrics (BLEU, METEOR, CIDEr)
python evaluate_caption_metrics.py

# Run full visualization suite
python run_visualizations.py
```

## 🧠 Technical Architecture

### ESN (Echo State Network)
- **Purpose**: Reservoir-based recurrent neural network for cultural signal extraction
- **Features**: 28-dimensional feature vector capturing cultural markers
- **Output**: Cultural mode classification with confidence scores

### GRU (Gated Recurrent Unit)
- **Purpose**: Deep learning-based cultural context modeling
- **Architecture**: 2-layer GRU with 128 hidden units
- **Training**: Learns cultural patterns from annotated samples

### Florence-2
- **Purpose**: Vision-to-language model for initial caption generation
- **Capabilities**: Detailed scene understanding and description
- **Input**: Images up to 512x512 resolution

### Ollama Integration
- **Model**: llama3.2:3b (3B parameter local LLM)
- **Purpose**: Cultural refinement, adaptation, and RAG-based fact grounding
- **Advantages**: Private, fast, GPU-accelerated

## 📊 Evaluation Metrics

The system supports multiple evaluation frameworks:

### Caption Quality Metrics
- **BLEU**: Bilingual Evaluation Understudy (precision-based)
- **METEOR**: Metric for Evaluation of Translation with Explicit Ordering
- **CIDEr**: Consensus-based Image Description Evaluation
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation

### Cultural Metrics
- **Cultural Density**: Quantifies cultural signal strength (0-1)
- **Heritage Alignment**: Validates traditional element presence
- **Surprise Detection**: Identifies unexpected cultural violations
- **Silence Metrics**: Detects meaningful absences in cultural signaling

## 🌐 Supported Languages

- English
- Hindi
- Urdu
- Tamil
- Telugu
- Kannada
- Malayalam
- Bengali
- Assamese
- Odia
- And 10+ more...

## 🎨 Features Explained

### Cultural Signal Dimensions
1. **Spice Level**: Intensity of traditional/cultural markers
2. **Ritual Signals**: Presence of ceremonial/sacred elements
3. **Daily Life**: Everyday routines and family moments
4. **Heritage**: Traditional/ancestral elements
5. **Color & Texture**: Visual cultural aesthetics
6. **Serving & Presentation**: Food preparation styles
7. **People & Community**: Social/communal aspects
8. **Offerings & Decorations**: Devotional elements

### Trust Analysis Dimensions
- **Vision Consistency**: Alignment between caption and classification
- **Model Coherence**: Agreement between ESN and GRU predictions
- **Signal Interpretation**: Internal consistency of cultural signals
- **Temporal Stability**: Consistency across multiple inferences

## 📈 Performance

Typical performance metrics:
- **Caption Generation**: ~2-5 seconds per image (CPU), <1 second (GPU)
- **Cultural Classification**: ~0.1 seconds
- **Trust Assessment**: ~0.2 seconds  
- **Full Analysis Pipeline**: ~5-10 seconds

## 🔄 Workflow

```
Input Image
    ↓
Florence-2 Vision Model → Base Caption
    ↓
ESN & GRU Classification → Cultural Mode + Confidence
    ↓
Signal Analysis → Feature Extraction
    ↓
Trust & Safety Assessment
    ↓
Cultural Refinement (Ollama LLM)
    ↓
Translation (if needed) → Multilingual Output
    ↓
Audio Generation (gTTS) + Database Caching
    ↓
Visualization & Analytics Output
```

## 🛠️ Advanced Configuration

### Training Custom Models

```python
from cultural_context import CulturalContextManager

manager = CulturalContextManager()

training_samples = [
    ("A traditional thali ceremony", "Traditional"),
    ("Diwali celebration with lights", "Festival"),
    ("Daily family dinner routine", "Daily Life"),
    ("A plate of food", "Generic"),
]

manager.train(training_samples)
```

### Custom Database Entries

```python
from app import add_cultural_entry

add_cultural_entry(
    mode="Traditional",
    name="Thali",
    description="A complete meal served on a platter",
    region="India-wide",
    festival="Diwali, Holi",
    story="Traditional way of serving meals...",
    recipe_summary="Multiple dishes in balanced portions"
)
```

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` in configuration files
- Use CPU mode if GPU memory is limited
- Clear HuggingFace cache: `rm -rf hf_cache/`

### Slow Caption Generation
- Enable GPU acceleration if available
- Pre-download models before first run
- Use smaller image sizes (512x512)

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check localhost:11434 is accessible
- Verify model is downloaded: `ollama list`

## 📚 References & Datasets

- **COCO Dataset**: Microsoft COCO evaluation toolkit included (`pycocoevalcap/`)
- **Image Evaluation Set**: Sample images in `eval/` directory  
- **Reference Captions**: See `references.json`
- **Cultural Database**: SQLite schema in `cultural_data.db`

## 📄 Databases & Outputs

### SQLite Schema (cultural_data.db)
- `cultural_explorer` table: Caches captions and signal vectors
- `cultural_db` table: Heritage, rituals, and cultural facts
- Enables efficient retrieval for RAG-based refinement

### Output Formats
- **JSON**: Structured results from `model_output.json`
- **CSV**: Evaluation metrics in `caption_metrics_results.csv`
- **PNG**: Visualization plots in `evaluation_outputs/`

## 🔐 Privacy & Safety

- **Local Processing**: Optional local Ollama LLM (no cloud calls)
- **No Data Persistence**: Temporary image files automatically cleaned
- **Cultural Sensitivity**: Built-in trust and safety analysis
- **Bias Detection**: Identifies surprising or unexpected outputs

## 🚀 Future Enhancements

- [ ] Multi-image captioning for narrative generation
- [ ] Real-time video captioning support
- [ ] Fine-tuned cultural models for specific regions
- [ ] Weakref-based cultural graph visualization
- [ ] Integration with more LLMs (Mistral, Llama 2, etc.)
- [ ] Mobile deployment support

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{culturally_aware_captioning_2025,
  title = {Multilingual and Culturally-Aware Image Captioning and Audio Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Mayur-cinderace/Multilingual-And-Culturally-Aware-Image-Captioning-and-Audio-Generation}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact & Support

- **Issues**: Open a GitHub issue for bugs and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [Your contact information]

## 🙏 Acknowledgments

- Florence-2 team for the vision model
- HuggingFace for transformer models and infrastructure
- Ollama for local LLM capabilities
- COCO dataset creators and evaluation toolkit maintainers
- Google Translate API for multilingual support

---

**Built with ❤️ for culturally-aware AI systems**
