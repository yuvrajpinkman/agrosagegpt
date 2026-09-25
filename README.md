# 🌱 AgroSageGPT: Multilingual Generative AI Digital Agronomist

AgroSageGPT is a unified digital agronomist platform that synthesizes:
1. **Computer Vision (Crop Disease Detection)** using Ultralytics YOLOv8
2. **IoT Soil Telemetry Simulation** (7-in-1 NPK, pH, Moisture, Temperature, EC) using Pandas
3. **Live Agro-Meteorological Weather Data** via OpenWeatherMap API (with realistic agrometeorological fallback)
4. **Cross-Domain Retrieval-Augmented Generation (RAG)** using ChromaDB and LangChain to deliver multilingual, farmer-friendly advisories.

---

## 📁 Project Structure

```
aerosagegpt/
├── app.py                      # Main Streamlit Frontend Application
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables
├── .env                        # Local configuration
├── test_modules.py             # Automated unit and integration verification suite
├── data/
│   └── agri_knowledge.txt      # Curated agronomic reference knowledge base
├── sample_data/
│   ├── sample_diseased_leaf.jpg# Test image (Early Blight on foliage)
│   └── sample_healthy_leaf.jpg # Test image (Healthy foliage)
├── src/
│   ├── __init__.py
│   ├── vision_module.py        # Ultralytics YOLOv8 detection & lesion assessment
│   ├── soil_mock.py            # 7-in-1 IoT soil sensor data generator & formatter
│   ├── weather_api.py          # OpenWeatherMap API integration & advisory
│   └── rag_engine.py           # ChromaDB vector store, RAG pipeline & multilingual LLM
└── venv/                       # Isolated virtual environment
```

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment
The virtual environment is located in the project root:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (already done by default) and optionally fill in your API keys:
```ini
LLM_API_KEY=your_openai_or_gemini_api_key_here
OPENWEATHER_API_KEY=your_openweather_key_here
DEFAULT_LOCATION=Bengaluru
```
> **Note**: AgroSageGPT is fully self-contained. If no API keys are provided, it seamlessly activates the built-in local agricultural synthesizer and realistic weather simulation so that you can test all features offline.

### 3. Launch the Streamlit Application
```powershell
.\venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Verification Suite
To verify all 4 core modules (Soil, Weather, Vision, ChromaDB RAG):
```powershell
.\venv\Scripts\python test_modules.py
```

---

## 🛠️ Features & Modules

### 📸 Tab 1: Crop Disease Vision
- Powered by **Ultralytics YOLOv8** (`yolov8n.pt`).
- Detects agricultural foliage objects and bounding boxes.
- Analyzes leaf lesion patterns to diagnose conditions such as **Early Blight (*Alternaria solani*)**, **Nutrient Chlorosis**, or **Healthy Foliage**.
- Includes 1-click sample loaders for instant testing.

### 🌦️ Tab 2: Real-time Weather
- Connects to OpenWeatherMap API for live temperature, humidity, wind speed, and pressure.
- Translates weather conditions into actionable agricultural advisories (e.g., spray drift precautions, fungal risk warnings).

### 🧪 Tab 3: IoT 7-in-1 Soil Telemetry
- Simulates real-time edge sensors measuring:
  - Nitrogen (N), Phosphorus (P), Potassium (K) in mg/kg
  - Soil pH (acidity / alkalinity)
  - Volumetric Soil Moisture (%)
  - Soil Temperature (°C)
  - Electrical Conductivity (EC in mS/cm)
- Visualizes 24-hour sensor trends using interactive line charts.
- Allows on-demand generation of new synthetic telemetry events.

### 💬 Tab 4: Multilingual Digital Agronomist Chat
- Ingests agronomic reference literature into a local **ChromaDB** vector store.
- Synthesizes the four pillars (Vision Diagnosis + Soil Telemetry + Weather Conditions + Knowledge Base) into a unified context prompt.
- Farmer-friendly 3-step action plans delivered in **English, Hindi (हिंदी), Kannada (ಕನ್ನಡ), Telugu (తెలుగు), or Tamil (தமிழ்)**.
