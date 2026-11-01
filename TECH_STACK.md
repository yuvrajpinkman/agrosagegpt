# AgroSageGPT: Complete Technical Stack & Architecture Specification

This document details the entire technology stack powering **AgroSageGPT** — an edge-compatible, multimodal Generative AI Digital Agronomist. Every framework, foundation model, Hugging Face checkpoint, cloud API, vector database, and utility module is documented below with exact identifiers, versions, and architectural roles.

---

## 1. System Architecture Overview

```
+--------------------------------------------------------------------------------------------------+
|                                    USER INTERFACE LAYER                                          |
|  Streamlit 1.64.0 (Light Mode) • Floating Dock (st.bottom) • Google Material Symbols Iconography  |
+--------------------------------------------------------------------------------------------------+
          |                                  |                                   |
          v                                  v                                   v
+-----------------------+          +-----------------------+           +-----------------------+
|  COMPUTER VISION DOCK |          |   AUDIO & VOICE DOCK  |           |   RAG & KNOWLEDGE     |
| • Meta DINOv2 (ViT)   |          | • Noise Gating Filter |           | • ChromaDB Vector DB  |
| • Gemini Multimodal   |          | • Gemini Flash ASR    |           | • PDF/TXT/MD Parser   |
| • YOLOv8 Spatial Loc  |          | • Groq Whisper-Turbo  |           | • Gemini Flash LLM    |
| • OpenCV HSV Lesion   |          | • Agri Lexicon Match  |           | • Groq Qwen LPU       |
+-----------------------+          | • Edge-TTS Neural     |           | • Dynamic Synthesizer |
                                   +-----------------------+           +-----------------------+
                                              |                                   |
                                              v                                   v
                                   +-----------------------------------------------------------+
                                   |                 TELEMETRY CONTEXT FUSION                  |
                                   | • 7-in-1 IoT Soil Sensor Sim (N, P, K, pH, Moisture, EC) |
                                   | • OpenWeatherMap Live API (Temp, Humidity, Wind Drift)    |
                                   +-----------------------------------------------------------+
```

---

## 2. Technology Stack Breakdown by Layer

### Layer A: Frontend & User Interface
| Component | Technology / Library | Version / Source | Exact Identifier / Role |
| :--- | :--- | :--- | :--- |
| **Framework** | **Streamlit** | `1.64.0` | Core reactive web application runtime (`app.py`). |
| **Theme / Design** | **Custom CSS** | Native HTML/CSS injection | High-contrast light mode (`#fafbfa` canvas, `#1b5e20` primary forest green, card shadows). |
| **Floating Dock** | `st.bottom` | Streamlit Native Proxy | Anchors floating input bar, camera flyout, microphone flyout, and document ingestion toolstrip at viewport bottom. |
| **Modal Flyouts** | `st.popover` | Streamlit Native Component | Zero-footprint flyouts for image capture, voice recording, manual upload, and prompt chips. |
| **Iconography** | **Google Material Symbols** | `:material/<name>:` | Clean vector icons (`photo_camera`, `mic`, `upload_file`, `lightbulb`, `smart_toy`, `person`, `sensors`, `biotech`). Replaced all informal emojis. |
| **Image Processing** | **Pillow (PIL)** | `10.x / 11.x` | In-memory image decoding, format conversions (RGB/BGR), and preview rendering. |

---

### Layer B: Computer Vision & Foliage Pathology Pipeline
AgroSageGPT uses a **3-tier hybrid vision architecture** balancing instant local classification with expert cloud multimodal reasoning.

| Tier | Component | Technology / Provider | Model Name / Hugging Face ID | Purpose & Implementation Details |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Local)** | **Primary Classifier** | Hugging Face Transformers (`5.22.1`) + PyTorch (`2.14.0+cpu`) | **`rodynaemad/plant-disease-dinov2-small`** | **Meta DINOv2 (Vision Transformer)** fine-tuned on the **PlantVillage** dataset.<br>• Covers **38 classes** (Apple Scab, Corn Blight, Potato Early/Late Blight, Tomato Early Blight, healthy leaves, etc.).<br>• Runs **100% locally / offline** on CPU in ~35ms with zero API cost. |
| **Tier 2 (Cloud)** | **Expert Multimodal Validator** | Google Generative AI Python SDK (`google-generativeai`) | **`gemini-3.6-flash`**<br>*(Failover Pool: `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-flash-latest`, `gemini-3.8-flash`)* | Evaluates leaf photo alongside HF DINOv2 predictions.<br>• Outputs structured JSON validation verdict (`Agreed`, `Refined`, `Corrected`).<br>• Pinpoints foliar morphology (concentric rings, chlorotic halos, necrotic centers) and prescribes specific chemical/biological dosages (Mancozeb, Chlorothalonil, Trichoderma). |
| **Spatial** | **Object Localization** | Ultralytics YOLOv8 (`8.4.14`) | **`yolov8n.pt`** | Detects and draws bounding boxes around crop foliage and plant bodies to count affected foliage clusters. |
| **Tier 3 (Offline)** | **Heuristic Fallback** | OpenCV (`opencv-python`) | `cv2.cvtColor(HSV)` | Color-space thresholding to calculate necrotic lesion pixel ratio if both network and local transformers are unavailable. |

---

### Layer C: Conversational RAG Engine & Language Models

| Sub-system | Technology / Provider | Model / API Endpoint | Purpose & Configuration |
| :--- | :--- | :--- | :--- |
| **Vector Database** | **ChromaDB** | Local `chromadb.PersistentClient` (`./.chroma_db`) | Stores embeddings of standard ICAR/agricultural manuals and dynamic user-uploaded PDF/TXT documents. |
| **Document Ingestion** | **pypdf** (`5.x`) | `pypdf.PdfReader` | Extracts raw text from uploaded crop bulletins, extension service manuals, and seed guides for real-time vectorization. |
| **Primary LLM** | **Google Gemini** | **`gemini-3.6-flash`**<br>*(Active Pool: `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-flash-latest`)* | Primary agronomist reasoning engine. Fuses live soil telemetry, weather metrics, visual pathology, and vector context into farmer-friendly advisories.<br>• Uses `request_options={"retry": None, "timeout": 8.0}` to prevent client hangs. |
| **Secondary LLM** | **Groq Cloud LPUs** | **`qwen/qwen3.8-27b`**<br>*(Failover: `llama-3.3-70b-versatile`)* | Ultra-fast LPU inference (under 400ms) with high multilingual and Indic language fluency when Gemini quotas are exhausted. |
| **Offline Synthesizer**| **Rule-Based Synthesis** | Native Python (`src/rag_engine.py`) | Context-aware fallback engine providing full nursery, sowing, fertilizer (Urea/DAP/MOP), and irrigation schedules for rice, wheat, tomato, and maize. |
| **Language Detector** | **Custom Hybrid Detector** | Regex + Unicode Ranges + Phonetic Lexicon | Detects **English**, **Hindi**, **Kannada**, **Telugu**, and **Tamil**.<br>• Checks Unicode script blocks (`\u0900-\u097F`, `\u0C80-\u0CFF`, `\u0C00-\u0C7F`, `\u0B80-\u0BFF`).<br>• Analyzes romanized Hinglish tokens (e.g. *"chawal kab aur kaise ugaye"* $\rightarrow$ Hindi). |

---

### Layer D: Speech-to-Text (STT) Audio Pipeline
Built to handle tractor engine rumble, diesel pump vibrations, and multi-speaker farm environments.

| Pipeline Stage | Technology | Exact Component / Model | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Noise Gating** | NumPy & SciPy Digital Signal Processing | Butterworth High-Pass / Sub-80Hz Band-stop filter | Strips low-frequency tractor engine thrum (<80Hz) and ambient wind shear before feeding audio to speech models. |
| **2. Primary ASR** | Google Generative AI | **`gemini-3.6-flash` / `gemini-3.5-flash`** | Native multimodal audio transcription across English, Hindi, and regional Indic languages. |
| **3. Failover ASR** | Groq Cloud | **`whisper-large-v3-turbo`** (OpenAI Whisper on Groq LPUs) | Sub-second speech recognition with automatic language identification (`language=None`). |
| **4. Word Restoration** | Custom Lexicon Engine (`src/audio_pipeline.py`) | Domain Lexicon + Phonetic String Distance | Corrects phonetic transcription errors in Indian crop names, fertilizers (DAP, MOP, Urea), and pesticides (Mancozeb, Imidacloprid, Trichoderma). |

---

### Layer E: Text-to-Speech (TTS) Voice Synthesis

| Priority | Engine | Provider / Library | Exact Voice Identifier | Target Language |
| :--- | :--- | :--- | :--- | :--- |
| **Primary (HD Neural)** | **Microsoft Edge Neural TTS** | `edge-tts` (Asynchronous Python client) | **`en-IN-NeerjaNeural`**<br>**`hi-IN-MadhurNeural`**<br>**`kn-IN-GaganNeural`**<br>**`te-IN-MohanNeural`**<br>**`ta-IN-PallaviNeural`** | English (India)<br>Hindi<br>Kannada<br>Telugu<br>Tamil |
| **Secondary (Web)** | **Google Translate TTS** | `gTTS` | Regional language voice streaming | Multi-language fallback |
| **Tertiary (Offline)** | **Meta MMS TTS** | Hugging Face Transformers | `facebook/mms-tts-hin`, `facebook/mms-tts-kan`, `facebook/mms-tts-tel`, `facebook/mms-tts-tam` | Completely local offline speech synthesis |

---

### Layer F: IoT Sensor Simulation & Live Weather

| Module | Source / API | Parameters Monitored | Implementation |
| :--- | :--- | :--- | :--- |
| **IoT Soil Simulation** | Synthetic 7-in-1 NPK Sensor (`src/soil_mock.py`) | Nitrogen (N), Phosphorus (P), Potassium (K) in mg/kg, Soil pH, Moisture (%), Temperature (°C), Electrical Conductivity (EC in mS/cm) | Generates a 24-hour diurnal rolling history with realistic agricultural fluctuations via `pandas.DataFrame`. |
| **Live Weather Station** | **OpenWeatherMap API** (`api.openweathermap.org/data/2.5/weather`) | Temperature (°C), Feels-like (°C), Humidity (%), Wind Speed (m/s), Atmospheric Conditions | Live API queries formatted into pesticide drift safety assessments (`wind_speed < 4.5 m/s`). Includes microclimate simulator fallback. |

---

## 3. Complete Dependencies & Version Manifest

The application runs on **Python 3.11.9 (64-bit)** with the following core packages:

```toml
# Core Web & UI
streamlit = "1.64.0"
pillow = "10.x"
pandas = "2.x"

# Machine Learning & Computer Vision
torch = "2.14.0+cpu"
transformers = "5.22.1"
ultralytics = "8.4.14"        # YOLOv8
opencv-python = "4.x"
safetensors = "0.8.0"
huggingface-hub = "1.33.0"

# Knowledge & Vector Database
chromadb = "0.5.x"
pypdf = "5.x"

# Cloud AI & Model Providers
google-generativeai = "0.8.x" # Google Gemini API
groq = "0.18.x"               # Groq Cloud LPUs
requests = "2.32.x"
python-dotenv = "1.0.x"

# Audio, Speech & Voice
edge-tts = "6.1.x"            # Microsoft Neural Voices
gTTS = "2.5.x"                # Google Translate TTS
```

---

## 4. Environment Variables (`.env`)

| Variable Name | Required | Service | Purpose |
| :--- | :---: | :--- | :--- |
| `GEMINI_API_KEY` | Yes | Google AI Studio | Powers Gemini Flash RAG, Gemini Multimodal Vision Validator, and primary ASR. |
| `GROQ_API_KEY` | Optional | Groq Cloud | Powers Whisper-Large-v3-Turbo speech transcription and Qwen 27B LPU fallback. |
| `OPENWEATHER_API_KEY` | Optional | OpenWeatherMap | Fetches live ambient temperature, humidity, and wind speed for spray safety. |
| `DEFAULT_LOCATION` | Optional | OpenWeatherMap | Default farm location (e.g. `Bengaluru`). |
