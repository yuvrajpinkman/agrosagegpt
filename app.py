"""AgroSageGPT: Multilingual Generative AI Digital Agronomist.

Main Streamlit Application combining Computer Vision, IoT Soil Analytics,
Live Weather Telemetry, and Cross-Domain Multilingual RAG.
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
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Namaste! I am **AgroSageGPT**, your Digital Agronomist. Upload a crop photo, inspect your live soil/weather telemetry, or ask me any question about crop health, fertilizer dosing, and pest management in your language."
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

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🌾 AgroSage Settings")
    
    language = st.selectbox(
        "🌐 Language / भाषा / ಭಾಷೆ / భాష / மொழி:",
        ["English", "Hindi", "Kannada", "Telugu", "Tamil"],
        index=0
    )
    
    crop_type = st.selectbox(
        "🌱 Primary Crop:",
        ["Tomato", "Rice / Paddy", "Maize / Corn", "Cotton", "Wheat", "Chili", "Potato"],
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
    st.markdown("### ⚙️ System Telemetry")
    st.success("✅ Ultralytics YOLOv8 Loaded")
    st.success("✅ ChromaDB RAG Vector Store Active")
    st.success("✅ 7-in-1 IoT Soil Telemetry Online")
    
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Chat history cleared. How can I assist you with your farm today?"}
        ]
        st.rerun()

# Header
st.markdown('<div class="main-header">🌱 AgroSageGPT: Multilingual Digital Agronomist</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unified Generative AI & RAG Platform combining Computer Vision, IoT Soil Analytics, and Agro-Meteorology</div>', unsafe_allow_html=True)

# Tabs
tab_vision, tab_weather, tab_soil, tab_chat = st.tabs([
    "📸 Crop Disease Vision",
    "🌦️ Real-time Weather",
    "🧪 IoT Soil Telemetry",
    "💬 Digital Agronomist Chat"
])

# ----------------- TAB 1: VISION MODULE -----------------
with tab_vision:
    st.subheader("📸 Crop & Leaf Lesion Diagnostic (Ultralytics YOLOv8)")
    st.caption("Upload a photo of your crop foliage or select a sample image to diagnose diseases, pests, and nutrient chlorosis.")
    
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
                with st.spinner("Analyzing image features with YOLOv8..."):
                    res = analyze_crop_image(target_img, crop_type=crop_type)
                    st.session_state.vision_result = res
                    st.success("Diagnosis Complete!")
                    
    with col_v2:
        if st.session_state.vision_result is not None:
            v_res = st.session_state.vision_result
            st.image(v_res["annotated_image"], caption="YOLOv8 Object & Lesion Localization", use_container_width=True)
            
            diag = v_res["disease_diagnosis"]
            severity_color = "badge-alert" if "High" in diag["severity"] else ("badge-warning" if "Mild" in diag["severity"] else "badge-optimal")
            
            st.markdown(f"### Assessment: **{diag['condition']}**")
            st.markdown(f"**Confidence:** `{diag['confidence'] * 100:.1f}%` | **Severity:** <span class='status-badge {severity_color}'>{diag['severity']}</span>", unsafe_allow_html=True)
            st.info(f"**Symptoms:** {diag['symptoms']}")
            st.warning(f"**Recommended Field Action:** {diag['recommendation']}")
        else:
            st.info("Upload or select an image on the left and click **'Run Computer Vision Diagnostic'** to view the lesion segmentation and classification report.")


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
    st.caption("Live edge telemetry streaming: Nitrogen (N), Phosphorus (P), Potassium (K), pH, Moisture, Temperature, EC.")
    
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
    with st.expander("🔍 View Context Injected into RAG Pipeline (Vision + Soil + Weather + ChromaDB)", expanded=False):
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
    for msg in st.session_state.chat_history:
        avatar = "🌾" if msg["role"] == "assistant" else "👨‍🌾"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
    # Quick suggestion buttons
    st.markdown("<small>💡 Quick Farmer Questions:</small>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_q = None
    if q_col1.button("🔍 What disease is on my leaves and how to treat it?", use_container_width=True):
        quick_q = "What disease is on my leaves and what spray should I apply?"
    if q_col2.button("🌱 Are my soil NPK and moisture levels healthy?", use_container_width=True):
        quick_q = "Are my soil NPK and moisture levels healthy for this crop?"
    if q_col3.button("🌦️ Is today's weather safe for spraying pesticide?", use_container_width=True):
        quick_q = "Is today's weather safe for spraying pesticide or should I delay?"

    # Chat Input
    user_prompt = st.chat_input("Type your farming question here...")
    final_query = quick_q or user_prompt
    
    if final_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": final_query})
        with st.chat_message("user", avatar="👨‍🌾"):
            st.markdown(final_query)
            
        # Prepare injected contexts
        v_summary = st.session_state.vision_result["summary"] if st.session_state.vision_result else "No image uploaded yet. Routine farm inspection."
        s_summary = st.session_state.soil_summary
        w_summary = st.session_state.weather_summary
        
        with st.chat_message("assistant", avatar="🌾"):
            with st.spinner("AgroSageGPT is reasoning over vision, soil telemetry, weather, and agronomic knowledge..."):
                response_data = ask_agronomist(
                    query=final_query,
                    vision_summary=v_summary,
                    soil_summary=s_summary,
                    weather_summary=w_summary,
                    language=language,
                    chat_history=st.session_state.chat_history
                )
                answer_text = response_data["answer"]
                st.markdown(answer_text)
                
                with st.expander("📚 Retrieved Agronomic Knowledge Chunks", expanded=False):
                    for idx, doc in enumerate(response_data.get("retrieved_context", [])):
                        st.markdown(f"**Source Document Chunk {idx+1}:**")
                        st.text(doc)
                        
        st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
