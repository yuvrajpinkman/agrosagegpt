"""AgroSageGPT: Multilingual Generative AI Digital Agronomist.

Modern Light-Theme Conversational UI with:
- Unified Chat Feed
- Downside Input Dock (Image Attachment, Voice Recording, Dynamic RAG Documents)
- Top Compact Telemetry Bar (Live Weather + 7-in-1 Soil Sensors)
- Dynamic On-The-Fly Document Ingestion into ChromaDB
- High-Definition Neural TTS & Gemini Flash ASR Pipeline
"""

import os
import io
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

# Streamlit Page Config - Light Mode Enforced
st.set_page_config(
    page_title="AgroSageGPT | Digital Agronomist",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Light-Mode CSS
st.markdown("""
<style>
    /* Global Light Background and Typography */
    .stApp {
        background-color: #fafbfa;
        color: #1a2e1a;
    }
    
    /* Top Header Bar */
    .top-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1rem;
        background: #ffffff;
        border-bottom: 1px solid #e1e8e1;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 40, 0, 0.04);
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        color: #4a6b4a;
    }
    
    /* Telemetry Ribbon Card */
    .telemetry-ribbon {
        background: #ffffff;
        border: 1px solid #e2ece2;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Modern Action Dock at downside */
    .action-dock {
        background: #ffffff;
        border: 1px solid #dce6dc;
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    
    /* Status tags & badges */
    .engine-tag {
        font-size: 0.76rem;
        color: #4b634b;
        background: #eef5ee;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 6px;
    }
    .badge-optimal { background-color: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;}
    .badge-alert { background-color: #ffebee; color: #c62828; padding: 2px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Namaste! I am **AgroSageGPT**, your Generative AI Digital Agronomist.\n\nYou can ask me questions about crop management, sowing schedules, fertilizer dosing, or plant health. Use the toolbar at the bottom to **attach leaf photos**, **speak via microphone**, or **upload custom agricultural manuals (PDF/TXT)**.",
            "engine": "AgroSage Welcome Agent",
            "audio_bytes": None,
            "attached_image": None
        }
    ]

if "soil_df" not in st.session_state:
    st.session_state.soil_df = generate_soil_history(hours=24)
    st.session_state.soil_dict = get_latest_soil_reading(st.session_state.soil_df, as_dict=True)
    st.session_state.soil_summary = get_latest_soil_reading(st.session_state.soil_df, as_dict=False)

if "weather_data" not in st.session_state:
    st.session_state.weather_data = get_weather_data()
    st.session_state.weather_summary = format_weather_summary(st.session_state.weather_data)

if "active_vision_summary" not in st.session_state:
    st.session_state.active_vision_summary = "No crop disease detected. General field management."

if "mic_key_idx" not in st.session_state:
    st.session_state.mic_key_idx = 0

if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []

# Sidebar for Preferences & Settings
with st.sidebar:
    st.markdown("### 🌾 Farm Configuration")
    
    language = st.selectbox(
        "🌐 Language / भाषा:",
        ["Hindi", "English", "Kannada", "Telugu", "Tamil"],
        index=0
    )
    
    crop_type = st.selectbox(
        "🌱 Primary Crop:",
        ["Rice / Paddy (धान)", "Wheat (गेहूं)", "Tomato (टमाटर)", "Maize / Corn (मक्का)", "Cotton (कपास)", "Chili (मिर्च)", "Potato (आलू)"],
        index=0
    )
    
    city_input = st.text_input("📍 Farm City:", value=os.getenv("DEFAULT_LOCATION", "Bengaluru"))
    if st.button("🔄 Sync Weather Location", use_container_width=True):
        st.session_state.weather_data = get_weather_data(city_input)
        st.session_state.weather_summary = format_weather_summary(st.session_state.weather_data)
        st.toast(f"Synced weather for {city_input}!", icon="🌤️")

    st.markdown("---")
    enable_tts = st.toggle("🔊 Read Answers Aloud (TTS)", value=True)
    
    st.markdown("---")
    st.markdown("### 📚 Custom Knowledge Base")
    st.caption("Documents dynamically ingested into ChromaDB:")
    if st.session_state.uploaded_doc_names:
        for doc_name in st.session_state.uploaded_doc_names:
            st.markdown(f"• 📄 `{doc_name}`")
    else:
        st.caption("*No custom files uploaded yet. Default agronomic KB active.*")
        
    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Chat cleared. What would you like to discuss about your crops today?", "engine": "AgroSage Reset", "audio_bytes": None, "attached_image": None}
        ]
        st.session_state.active_vision_summary = "No crop disease detected. General field management."
        st.rerun()

# ----------------- TOP BRANDING & TELEMETRY BAR -----------------
st.markdown("""
<div class="top-brand">
    <div>
        <div class="brand-title">🌱 AgroSageGPT</div>
        <div class="brand-subtitle">Multilingual Generative AI Digital Agronomist • Light Edition</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Compact Horizontal Telemetry Ribbon
w = st.session_state.weather_data
s = st.session_state.soil_dict

with st.expander(f"📊 Live Field Telemetry ({w.get('city', 'Farm')}) • 🌡️ {w['temperature_c']}°C | 💧 Humidity {w['humidity_pct']}% | 🧪 NPK {s['nitrogen']['value']}-{s['phosphorus']['value']}-{s['potassium']['value']} | ⚖️ pH {s['ph']['value']}", expanded=False):
    t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns(5)
    t_c1.metric("Temperature", f"{w['temperature_c']} °C", f"Feels {w['feels_like_c']} °C")
    t_c2.metric("Humidity", f"{w['humidity_pct']} %", w['condition'])
    t_c3.metric("Wind Speed", f"{w['wind_speed_ms']} m/s", "Drift safe" if w['wind_speed_ms'] < 4.5 else "High drift")
    t_c4.metric("NPK Level", f"{s['nitrogen']['value']}-{s['phosphorus']['value']}-{s['potassium']['value']}", "Optimal" if s['nitrogen']['status'] == "Medium" else "Review")
    t_c5.metric("Soil Moisture", f"{s['moisture']['value']} %", s['moisture']['status'])
    
    st.caption(f"Weather Source: {w.get('source')} | Soil: 7-in-1 IoT Telemetry")
    if st.button("📡 Simulate New Soil Telemetry Event", type="secondary"):
        st.session_state.soil_df = generate_soil_history(hours=24)
        st.session_state.soil_dict = get_latest_soil_reading(st.session_state.soil_df, as_dict=True)
        st.session_state.soil_summary = get_latest_soil_reading(st.session_state.soil_df, as_dict=False)
        st.rerun()

# ----------------- CHAT CONVERSATION FEED -----------------
chat_container = st.container()

with chat_container:
    for idx, msg in enumerate(st.session_state.chat_history):
        avatar = "🌾" if msg["role"] == "assistant" else "👨‍🌾"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("attached_image") is not None:
                st.image(msg["attached_image"], caption="Attached Crop Foliage", width=300)
            
            st.markdown(msg["content"])
            
            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format="audio/mp3")
                
            if msg.get("engine"):
                st.markdown(f"<span class='engine-tag'>⚡ Powered by: {msg['engine']}</span>", unsafe_allow_html=True)

# ----------------- DOWNSIDE ATTACHMENT & ACTION DOCK -----------------
st.markdown("---")

dock_tab1, dock_tab2, dock_tab3 = st.tabs([
    "📎 Attach Image (Leaf / Crop Diagnosis)",
    "🎙️ Voice Microphone (Speak in your language)",
    "📄 Upload Custom Knowledge (PDF / TXT Manuals)"
])

# Dock Tab 1: Image Attachment
staged_image = None
with dock_tab1:
    img_col1, img_col2 = st.columns([2, 1])
    with img_col1:
        uploaded_img = st.file_uploader("Upload leaf / plant image for disease detection", type=["jpg", "jpeg", "png"], key="dock_img_uploader")
    with img_col2:
        st.write("Or quick samples:")
        sample_btns = st.columns(2)
        use_diseased = sample_btns[0].button("🍃 Diseased Leaf", use_container_width=True)
        use_healthy = sample_btns[1].button("🌿 Healthy Leaf", use_container_width=True)
        
    if uploaded_img is not None:
        staged_image = Image.open(uploaded_img)
    elif use_diseased and os.path.exists("sample_data/sample_diseased_leaf.jpg"):
        staged_image = Image.open("sample_data/sample_diseased_leaf.jpg")
    elif use_healthy and os.path.exists("sample_data/sample_healthy_leaf.jpg"):
        staged_image = Image.open("sample_data/sample_healthy_leaf.jpg")
        
    if staged_image is not None:
        st.image(staged_image, caption="Ready to analyze with question", width=180)
        st.caption("✅ Image attached. Type your question below or click 'Analyze Image Now'.")
        if st.button("🔍 Analyze Image & Diagnose Plant Health", type="primary"):
            with st.spinner("Diagnosing foliage with Gemini Multimodal Vision & YOLO..."):
                v_res = analyze_crop_image(staged_image, crop_type=crop_type)
                diag = v_res["disease_diagnosis"]
                st.session_state.active_vision_summary = v_res["summary"]
                
                # Formulate assistant diagnostic response
                diag_msg = (
                    f"### 🩺 Plant Pathology Assessment: **{diag['condition']}**\n\n"
                    f"- **Severity:** `{diag['severity']}` (Confidence: {diag['confidence']*100:.1f}%)\n"
                    f"- **Observed Symptoms:** {diag['symptoms']}\n"
                    f"- **Recommended Field Action:** {diag['recommendation']}\n\n"
                    f"*Diagnosed by: {diag.get('source', 'Vision Engine')}*"
                )
                
                audio_bytes = None
                if enable_tts:
                    audio_bytes, _ = synthesize_speech(f"{diag['condition']}. {diag['recommendation']}", language=language)
                    
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"Please diagnose this {crop_type} crop foliage photo.",
                    "attached_image": staged_image,
                    "audio_bytes": None,
                    "engine": None
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": diag_msg,
                    "attached_image": v_res.get("annotated_image"),
                    "audio_bytes": audio_bytes,
                    "engine": diag.get("source", "Gemini Vision")
                })
                st.rerun()

# Dock Tab 2: Voice Input (Microphone with Re-record)
voice_text_query = None
with dock_tab2:
    v_head1, v_head2 = st.columns([3, 1])
    with v_head1:
        st.markdown(f"**Speak your question in {language}**")
    with v_head2:
        if st.button("🔄 Record Again", use_container_width=True, help="Clear microphone and record a fresh question"):
            st.session_state.mic_key_idx += 1
            st.rerun()
            
    mic_audio = st.audio_input(
        f"🎙️ Click to speak ({language})",
        key=f"bottom_dock_mic_{st.session_state.mic_key_idx}"
    )
    if mic_audio is not None:
        with st.spinner(f"Transcribing voice using Gemini Flash ASR ({language})..."):
            t_text, t_engine = transcribe_audio(mic_audio, language=language)
            if t_text and len(t_text.strip()) > 3:
                voice_text_query = t_text
                st.info(f"🗣️ **Voice Recognized:** *\"{voice_text_query}\"*")
                if st.button("🚀 Ask Recognized Question", type="primary"):
                    # Will trigger execution below
                    pass

# Dock Tab 3: Dynamic Knowledge Document Ingestion (PDF / TXT / MD)
with dock_tab3:
    st.markdown("**Upload Your Custom Agricultural Manuals, University Bulletins, or Seed Guides**")
    st.caption("Documents uploaded here are dynamically parsed and indexed into ChromaDB vector store on the fly.")
    uploaded_doc = st.file_uploader("Upload agricultural file (.pdf, .txt, .md)", type=["pdf", "txt", "md"], key="dynamic_rag_uploader")
    if uploaded_doc is not None:
        if uploaded_doc.name not in st.session_state.uploaded_doc_names:
            with st.spinner(f"Extracting and indexing `{uploaded_doc.name}` into ChromaDB..."):
                doc_bytes = uploaded_doc.read()
                kb = get_knowledge_store()
                num_chunks = kb.ingest_user_document(doc_bytes, uploaded_doc.name)
                st.session_state.uploaded_doc_names.append(uploaded_doc.name)
                st.success(f"✅ Successfully ingested **{num_chunks} knowledge chunks** from `{uploaded_doc.name}` into active RAG memory!")
                st.caption("You can now ask questions directly about this document!")

# Quick suggestion chips
chip1, chip2, chip3 = st.columns(3)
quick_chip_query = None
if chip1.button("🌾 धान (चावल) कब और कैसे उगाएं?", use_container_width=True):
    quick_chip_query = "धान (चावल) कब और कैसे उगाएं? नर्सरी से कटाई तक पूरी विधि बताएं।"
if chip2.button("🌱 क्या मेरी मिट्टी में खाद की जरूरत है?", use_container_width=True):
    quick_chip_query = "मेरी मिट्टी के NPK, pH और नमी के अनुसार कौन सी खाद देनी चाहिए?"
if chip3.button("🌦️ क्या आज कीटनाशक छिड़काव के लिए मौसम सही है?", use_container_width=True):
    quick_chip_query = "आज के मौसम, हवा और तापमान के अनुसार क्या कीटनाशक छिड़काव करना सुरक्षित है?"

# Main Chat Input Box at Bottom
user_text_input = st.chat_input("Type your farming question here (e.g. चावल कब और कैसे उगाएं)...")

# Resolve Query Priority: Typed Text > Voice Query > Quick Suggestion Chips
final_query = user_text_input or voice_text_query or quick_chip_query

if final_query:
    last_user_msg = None
    for m in reversed(st.session_state.chat_history):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
            
    if final_query != last_user_msg:
        # Append User Message
        st.session_state.chat_history.append({
            "role": "user",
            "content": final_query,
            "attached_image": staged_image,
            "audio_bytes": None,
            "engine": None
        })
        
        # Prepare context inputs
        v_summary = st.session_state.active_vision_summary
        s_summary = st.session_state.soil_summary
        w_summary = st.session_state.weather_summary
        
        with st.chat_message("assistant", avatar="🌾"):
            with st.spinner("AgroSageGPT is synthesizing your agronomic guidance..."):
                response_data = ask_agronomist(
                    query=final_query,
                    vision_summary=v_summary,
                    soil_summary=s_summary,
                    weather_summary=w_summary,
                    language=language,
                    chat_history=st.session_state.chat_history
                )
                answer_text = response_data["answer"]
                engine_used = response_data.get("engine", "AgroSage Engine")
                
                st.markdown(answer_text)
                
                # Synthesize High-Definition Neural Audio
                audio_bytes = None
                if enable_tts:
                    with st.spinner(f"Synthesizing neural voice in {language}..."):
                        audio_bytes, tts_engine_name = synthesize_speech(answer_text, language=language)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            st.caption(f"🔊 Spoken by: {tts_engine_name}")
                            
                st.markdown(f"<span class='engine-tag'>⚡ Powered by: {engine_used}</span>", unsafe_allow_html=True)
                
                with st.expander("📚 Retrieved Agronomic Knowledge Chunks", expanded=False):
                    for idx, doc in enumerate(response_data.get("retrieved_context", [])):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.text(doc)
                        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "attached_image": None,
            "engine": engine_used,
            "audio_bytes": audio_bytes
        })
        st.rerun()
