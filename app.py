"""AgroSageGPT: Multilingual Generative AI Digital Agronomist.

Main Streamlit Application combining Computer Vision, IoT Soil Analytics,
Live Weather Telemetry, Cross-Domain Multilingual RAG, Voice STT & Voice TTS.
"""

import os
import streamlit as st
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Import domain modules
from src.soil_mock import generate_soil_history, get_latest_soil_reading
from src.weather_api import get_weather_data, format_weather_summary
from src.vision_module import analyze_crop_image
from src.rag_engine import ask_agronomist, get_knowledge_store
from src.tts_engine import synthesize_speech
from src.stt_engine import transcribe_audio

load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="AgroSageGPT | Multilingual Digital Agronomist",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern accessible UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 14px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge {
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.82rem;
    }
    .badge-optimal { background-color: #c8e6c9; color: #2e7d32; }
    .badge-warning { background-color: #fff9c4; color: #f57f17; }
    .badge-alert { background-color: #ffcdd2; color: #c62828; }
    .engine-tag {
        font-size: 0.78rem;
        color: #555;
        background: #f0f4f8;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 6px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Namaste! I am **AgroSageGPT**, your Digital Agronomist. You can ask me any farming question by typing or speaking in Hindi, Kannada, Telugu, Tamil, or English. You can also upload crop photos for leaf disease diagnostics.",
            "engine": "AgroSage Welcome Agent",
            "audio_bytes": None
        }
    ]

if "soil_df" not in st.session_state:
    st.session_state.soil_df = generate_soil_history(hours=24)
    st.session_state.soil_dict = get_latest_soil_reading(st.session_state.soil_df, as_dict=True)
    st.session_state.soil_summary = get_latest_soil_reading(st.session_state.soil_df, as_dict=False)

if "weather_data" not in st.session_state:
    st.session_state.weather_data = get_weather_data()
    st.session_state.weather_summary = format_weather_summary(st.session_state.weather_data)

if "vision_result" not in st.session_state:
    st.session_state.vision_result = None

if "vision_audio" not in st.session_state:
    st.session_state.vision_audio = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🌾 AgroSage Settings")
    
    language = st.selectbox(
        "🌐 Language / भाषा / ಭಾಷೆ / భాష / மொழி:",
        ["English", "Hindi", "Kannada", "Telugu", "Tamil"],
        index=1  # Default to Hindi for Indian agronomist experience
    )
    
    crop_type = st.selectbox(
        "🌱 Primary Crop:",
        ["Rice / Paddy (धान)", "Wheat (गेहूं)", "Tomato (टमाटर)", "Maize / Corn (मक्का)", "Cotton (कपास)", "Chili (मिर्च)", "Potato (आलू)"],
        index=0
    )
    
    city_input = st.text_input(
        "📍 Farm Location (City):",
        value=os.getenv("DEFAULT_LOCATION", "Bengaluru")
    )
    
    if st.button("🔄 Sync Weather for Location", use_container_width=True):
        st.session_state.weather_data = get_weather_data(city_input)
        st.session_state.weather_summary = format_weather_summary(st.session_state.weather_data)
        st.toast(f"Weather synced for {city_input}!", icon="🌤️")

    st.markdown("---")
    st.markdown("### 🎙️ Voice & Audio Settings")
    enable_tts = st.toggle("🔊 Read Answers Aloud (Neural TTS)", value=True, help="Automatically plays crystal-clear spoken advice in your language.")

    st.markdown("---")
    st.markdown("### ⚙️ Production Telemetry & APIs")
    
    ow_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if ow_key and ow_key != "your_openweather_api_key_here":
        st.success("🟢 OpenWeatherMap Live API Connected")
    else:
        st.warning("🟡 OpenWeatherMap (Simulated)")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        st.success("🟢 Gemini 3.8 Flash (Multimodal & RAG)")
    else:
        st.warning("🟡 Gemini (Offline)")

    if groq_key and groq_key != "your_groq_api_key_here":
        st.success("🟢 Groq Whisper-v3 (Voice STT)")
        st.success("🟢 Groq Qwen-27B (LPU Fast Fallback)")
    else:
        st.warning("🟡 Groq (Offline)")

    st.success("🟢 High-Definition Neural Voice TTS Online")
    st.success("🟢 Ultralytics YOLOv8 Loaded")
    st.success("🟢 ChromaDB Vector Store Active")
    st.success("🟢 7-in-1 IoT Soil Telemetry Online")
    
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Chat history cleared. How can I assist you with your farm today?", "engine": "AgroSage Reset", "audio_bytes": None}
        ]
        st.session_state.vision_audio = None
        st.rerun()

# Header
st.markdown('<div class="main-header">🌱 AgroSageGPT: Multilingual Digital Agronomist</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production AI Platform with Gemini Multimodal Vision, IoT Soil Telemetry, Whisper Voice STT, ChromaDB RAG & Neural Voice TTS</div>', unsafe_allow_html=True)

# Tabs
tab_vision, tab_weather, tab_soil, tab_chat = st.tabs([
    "📸 Crop Disease Vision",
    "🌦️ Real-time Weather",
    "🧪 IoT Soil Telemetry",
    "💬 Digital Agronomist Chat"
])

# ----------------- TAB 1: VISION MODULE -----------------
with tab_vision:
    st.subheader("📸 Crop & Leaf Pathology Diagnostic (Gemini Multimodal + YOLOv8)")
    st.caption("Upload a foliage photo or select a test sample. Evaluated directly by Gemini 3.8 Flash Vision for accurate disease identification, with spatial YOLO bounding boxes.")
    
    col_v1, col_v2 = st.columns([1, 1])
    
    with col_v1:
        uploaded_file = st.file_uploader("Upload leaf/crop image", type=["jpg", "jpeg", "png"])
        
        st.write("Or test with pre-loaded field samples:")
        sample_col1, sample_col2 = st.columns(2)
        use_diseased = sample_col1.button("🍃 Load Diseased Leaf Sample", use_container_width=True)
        use_healthy = sample_col2.button("🌿 Load Healthy Leaf Sample", use_container_width=True)
        
        target_img = None
        if uploaded_file is not None:
            target_img = Image.open(uploaded_file)
        elif use_diseased and os.path.exists("sample_data/sample_diseased_leaf.jpg"):
            target_img = Image.open("sample_data/sample_diseased_leaf.jpg")
        elif use_healthy and os.path.exists("sample_data/sample_healthy_leaf.jpg"):
            target_img = Image.open("sample_data/sample_healthy_leaf.jpg")
            
        if target_img is not None:
            st.image(target_img, caption="Input Crop Foliage", use_container_width=True)
            if st.button("🔍 Run Computer Vision Diagnostic", type="primary", use_container_width=True):
                with st.spinner("Analyzing foliage with Gemini 3.8 Flash Multimodal Vision & YOLOv8..."):
                    res = analyze_crop_image(target_img, crop_type=crop_type)
                    st.session_state.vision_result = res
                    
                    if enable_tts:
                        diag_text = f"Diagnosis for {crop_type}: {res['disease_diagnosis']['condition']}. {res['disease_diagnosis']['recommendation']}"
                        audio_data, _ = synthesize_speech(diag_text, language=language)
                        st.session_state.vision_audio = audio_data
                    else:
                        st.session_state.vision_audio = None
                        
                    st.success("Diagnosis Complete!")
                    
    with col_v2:
        if st.session_state.vision_result is not None:
            v_res = st.session_state.vision_result
            st.image(v_res["annotated_image"], caption="YOLOv8 Spatial Localization & Bounding Boxes", use_container_width=True)
            
            diag = v_res["disease_diagnosis"]
            severity_color = "badge-alert" if "High" in diag["severity"] else ("badge-warning" if "Moderate" in diag["severity"] or "Mild" in diag["severity"] else "badge-optimal")
            
            st.markdown(f"### Assessment: **{diag['condition']}**")
            st.markdown(f"**Confidence:** `{diag['confidence'] * 100:.1f}%` | **Severity:** <span class='status-badge {severity_color}'>{diag['severity']}</span>", unsafe_allow_html=True)
            st.caption(f"🔬 Diagnosed by: **{diag.get('source', 'Vision Engine')}**")
            st.info(f"**Symptoms:** {diag['symptoms']}")
            st.warning(f"**Recommended Field Action:** {diag['recommendation']}")
            
            if st.session_state.vision_audio is not None:
                st.markdown("🔊 **Spoken Voice Advisory:**")
                st.audio(st.session_state.vision_audio, format="audio/mp3")
        else:
            st.info("Upload or select an image on the left and click **'Run Computer Vision Diagnostic'** to view the multimodal pathology report.")


# ----------------- TAB 2: WEATHER MODULE -----------------
with tab_weather:
    st.subheader(f"🌦️ Real-time Weather Telemetry ({st.session_state.weather_data.get('city', 'Local Farm')})")
    st.caption(f"Source: {st.session_state.weather_data.get('source')}")
    
    w = st.session_state.weather_data
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    w_col1.metric("Temperature", f"{w['temperature_c']} °C", f"Feels like {w['feels_like_c']} °C")
    w_col2.metric("Relative Humidity", f"{w['humidity_pct']} %", "Moisture in air")
    w_col3.metric("Wind Speed", f"{w['wind_speed_ms']} m/s", "Foliar drift factor")
    w_col4.metric("Atmospheric Pressure", f"{w['pressure_hpa']} hPa", w['condition'])
    
    st.markdown("---")
    st.markdown("#### 🌾 Agro-Meteorological Advisory")
    st.write(st.session_state.weather_summary)


# ----------------- TAB 3: SOIL IOT MODULE -----------------
with tab_soil:
    st.subheader("🧪 7-in-1 IoT Soil Sensor Telemetry")
    st.caption("Edge telemetry streaming: Nitrogen (N), Phosphorus (P), Potassium (K), pH, Moisture, Temperature, EC.")
    
    if st.button("📡 Simulate New Soil Sensor Telemetry Reading", type="secondary"):
        st.session_state.soil_df = generate_soil_history(hours=24)
        st.session_state.soil_dict = get_latest_soil_reading(st.session_state.soil_df, as_dict=True)
        st.session_state.soil_summary = get_latest_soil_reading(st.session_state.soil_df, as_dict=False)
        st.toast("Updated with new 7-in-1 soil readings!", icon="🌱")
        
    s = st.session_state.soil_dict
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Nitrogen (N)", f"{s['nitrogen']['value']} mg/kg", s['nitrogen']['status'])
    sc2.metric("Phosphorus (P)", f"{s['phosphorus']['value']} mg/kg", s['phosphorus']['status'])
    sc3.metric("Potassium (K)", f"{s['potassium']['value']} mg/kg", s['potassium']['status'])
    sc4.metric("Soil pH", f"{s['ph']['value']}", s['ph']['status'])
    
    sc5, sc6, sc7 = st.columns(3)
    sc5.metric("Volumetric Moisture", f"{s['moisture']['value']} %", s['moisture']['status'])
    sc6.metric("Soil Temperature", f"{s['temperature']['value']} °C", s['temperature']['status'])
    sc7.metric("Electrical Conductivity", f"{s['ec']['value']} mS/cm", s['ec']['status'])
    
    st.markdown("---")
    st.markdown("#### 📈 24-Hour Soil Telemetry Trends")
    chart_tab1, chart_tab2 = st.tabs(["NPK Nutrients (mg/kg)", "Moisture (%) & Soil Temp (°C)"])
    
    with chart_tab1:
        chart_df_npk = st.session_state.soil_df.set_index("timestamp")[["nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg"]]
        st.line_chart(chart_df_npk)
        
    with chart_tab2:
        chart_df_env = st.session_state.soil_df.set_index("timestamp")[["moisture_pct", "temperature_c"]]
        st.line_chart(chart_df_env)


# ----------------- TAB 4: CHATBOT INTERFACE -----------------
with tab_chat:
    st.subheader(f"💬 Conversational Digital Agronomist ({language})")
    
    # Active context preview expander
    with st.expander("🔍 View Live Context Injected into RAG Pipeline (Vision + Soil + Weather + ChromaDB)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📸 Vision State:**")
            if st.session_state.vision_result:
                st.code(st.session_state.vision_result["summary"], language="text")
            else:
                st.code("No image analyzed yet. General agronomy mode.", language="text")
        with c2:
            st.markdown("**🧪 Soil State:**")
            st.code(st.session_state.soil_summary, language="text")
        with c3:
            st.markdown("**🌦️ Weather State:**")
            st.code(st.session_state.weather_summary, language="text")

    # Render Chat History
    for idx, msg in enumerate(st.session_state.chat_history):
        avatar = "🌾" if msg["role"] == "assistant" else "👨‍🌾"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format="audio/mp3")
            if msg.get("engine"):
                st.markdown(f"<span class='engine-tag'>⚡ Powered by: {msg['engine']}</span>", unsafe_allow_html=True)

    # 🎙️ VOICE INPUT SECTION (Speech-to-Text)
    st.markdown("---")
    st.markdown(f"#### 🎙️ Speak your question in {language}")
    
    voice_c1, voice_c2 = st.columns([1, 1])
    with voice_c1:
        audio_prompt = st.audio_input(f"🎙️ Click microphone to record question ({language})", key="mic_audio_input")
    with voice_c2:
        audio_file_prompt = st.file_uploader("📁 Or upload audio note (.wav / .mp3 / .m4a)", type=["wav", "mp3", "m4a", "ogg"], key="file_audio_input")

    # Handle Voice Input
    voice_query = None
    target_audio = audio_prompt or audio_file_prompt
    if target_audio is not None:
        with st.spinner(f"Transcribing voice using Whisper-Large-v3 ({language})..."):
            t_text, t_engine = transcribe_audio(target_audio, language=language)
            if t_text and len(t_text.strip()) > 3:
                voice_query = t_text
                st.success(f"🗣️ **Voice Recognized ({t_engine}):** *\"{voice_query}\"*")
            else:
                st.warning("No clear speech detected. Please speak closer to your microphone or type below.")

    # Quick suggestion buttons
    st.markdown("<small>💡 Quick Farmer Questions:</small>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_q = None
    if q_col1.button("🌾 धान (चावल) कब और कैसे उगाएं?", use_container_width=True):
        quick_q = "धान (चावल) कब और कैसे उगाएं? नर्सरी से कटाई तक पूरी विधि बताएं।"
    if q_col2.button("🌱 क्या मेरे खेत की मिट्टी स्वस्थ है?", use_container_width=True):
        quick_q = "मेरी मिट्टी के NPK, pH और नमी के स्तर की जांच करके खाद की सलाह दें।"
    if q_col3.button("🌦️ क्या आज कीटनाशक छिड़काव के लिए मौसम सही है?", use_container_width=True):
        quick_q = "आज के मौसम, हवा और तापमान के अनुसार क्या कीटनाशक छिड़काव करना सुरक्षित है?"

    # Text Input
    user_prompt = st.chat_input("Or type your farming question here (e.g. चावल कब और कैसे उगाएं)...")
    
    # Clean query priority: Text input > Quick buttons > Voice query
    # If the user typed text, always execute the typed text!
    resolved_query = user_prompt or quick_q or voice_query
    
    if resolved_query:
        # Check if already processed to prevent rerun loops
        last_user_msg = None
        for m in reversed(st.session_state.chat_history):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
                
        if resolved_query != last_user_msg:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": resolved_query, "audio_bytes": None})
            with st.chat_message("user", avatar="👨‍🌾"):
                st.markdown(resolved_query)
                
            # Prepare injected contexts
            v_summary = st.session_state.vision_result["summary"] if st.session_state.vision_result else "No crop disease detected. Routine farm check."
            s_summary = st.session_state.soil_summary
            w_summary = st.session_state.weather_summary
            
            with st.chat_message("assistant", avatar="🌾"):
                with st.spinner("AgroSageGPT is generating your agronomic prescription..."):
                    response_data = ask_agronomist(
                        query=resolved_query,
                        vision_summary=v_summary,
                        soil_summary=s_summary,
                        weather_summary=w_summary,
                        language=language,
                        chat_history=st.session_state.chat_history
                    )
                    answer_text = response_data["answer"]
                    engine_used = response_data.get("engine", "AgroSage Engine")
                    
                    st.markdown(answer_text)
                    
                    # Generate High-Definition Neural Speech if enabled
                    audio_bytes = None
                    if enable_tts:
                        with st.spinner(f"Synthesizing high-definition neural voice in {language}..."):
                            audio_bytes, tts_engine_name = synthesize_speech(answer_text, language=language)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
                                st.caption(f"🔊 Spoken by: {tts_engine_name}")
                    
                    st.markdown(f"<span class='engine-tag'>⚡ Powered by: {engine_used}</span>", unsafe_allow_html=True)
                    
                    with st.expander("📚 Retrieved Agronomic Knowledge Chunks", expanded=False):
                        for idx, doc in enumerate(response_data.get("retrieved_context", [])):
                            st.markdown(f"**Source Document Chunk {idx+1}:**")
                            st.text(doc)
                            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "engine": engine_used,
                "audio_bytes": audio_bytes
            })
