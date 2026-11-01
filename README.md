# 🌱 AgroSageGPT: Multilingual Generative AI Digital Agronomist

AgroSageGPT is a unified digital agronomist platform that synthesizes:
1. **Multimodal Crop Vision**: Plant pathology diagnostics powered by Google Gemini 3.8 Flash Vision and spatial YOLOv8 object localization.
2. **Speech-to-Text (Voice Queries)**: Direct spoken voice input for rural farmers powered by Whisper-Large-v3-Turbo on Groq.
3. **Text-to-Speech (Spoken Advisories)**: Multilingual audio readout in regional languages (Hindi, Kannada, Telugu, Tamil, and English) using Neural TTS and Hugging Face Meta MMS.
4. **IoT Soil Telemetry**: 7-in-1 NPK, pH, Moisture, Temperature, and Electrical Conductivity (EC) real-time sensor analytics.
5. **Live Agro-Meteorology**: Real-time atmospheric weather data from OpenWeatherMap for spray timing and fungal disease risk calculation.
6. **Cross-Domain Multilingual RAG**: ChromaDB vector store coupled with Gemini 3.8 Flash (Primary) and Groq LPU LLM (High-Speed Failover).

---

## 📁 Project Structure

```
aerosagegpt/
├── app.py                      # Main Streamlit Frontend Application (Voice & Visuals)
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables
├── .env                        # Local configuration (protected from git)
├── test_modules.py             # 6-module automated unit and integration verification suite
├── data/
│   └── agri_knowledge.txt      # Curated agronomic reference knowledge base
├── sample_data/
│   ├── sample_diseased_leaf.jpg# Test image (Early Blight on foliage)
│   └── sample_healthy_leaf.jpg # Test image (Healthy foliage)
├── src/
│   ├── __init__.py
│   ├── vision_module.py        # Gemini Multimodal Vision + YOLOv8 localization
│   ├── soil_mock.py            # 7-in-1 IoT soil sensor data generator & formatter
│   ├── weather_api.py          # OpenWeatherMap API integration & advisory
│   ├── rag_engine.py           # ChromaDB vector store, Gemini 3.8 Flash & Groq RAG
│   ├── stt_engine.py           # Whisper-Large-v3 Speech-to-Text voice module
│   └── tts_engine.py           # Multilingual Text-to-Speech voice synthesis module
└── venv/                       # Isolated virtual environment
```

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
DEFAULT_LOCATION=Bengaluru
```

### 3. Launch the Application
```powershell
.\venv\Scripts\streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### 4. Run the Verification Suite
To verify all 6 modules (Soil, Weather, Vision, ChromaDB RAG, TTS, and STT):
```powershell
.\venv\Scripts\python test_modules.py
```

---

## 🛠️ Features & Architecture

### 🎙️ Voice-First Farmer Interaction (STT & TTS)
- **Speech-to-Text (STT)**: Farmers can click the microphone icon and speak in Hindi, Kannada, Telugu, Tamil, or English. Transcribed via Groq Whisper-Large-v3 in sub-seconds.
- **Text-to-Speech (TTS)**: Every agronomic advisory and plant diagnostic can be read aloud in the farmer's native tongue.

### 📸 Multimodal Crop Vision
- **Gemini 3.8 Flash Vision**: Evaluates uploaded foliage images for exact fungal, bacterial, viral lesions, nutrient deficiencies (chlorosis), or pest damage.
- **YOLOv8 Object Localization**: Detects leaf boundaries and spatial targets.

### 🌦️ Real-time Weather Telemetry
- Connects to OpenWeatherMap for live temperature, humidity, wind drift, and barometric pressure.
- Calculates safe pesticide application windows and fungal sporulation risk.

### 🧪 7-in-1 IoT Soil Sensor Telemetry
- Telemetry for Nitrogen (N), Phosphorus (P), Potassium (K), pH, Moisture, Temperature, and EC with 24-hour historical trends and agronomic status indicators.

### 💬 Multilingual Conversational RAG
- Dual-tier LLM architecture: **Google Gemini 3.8 Flash** as primary engine with automatic failover to **Groq (`openai/gpt-oss-120b`)** for zero-downtime resilience.
