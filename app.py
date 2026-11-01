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
from src.rag_engine import ask_agronomist, get_knowledge_store, detect_language
from src.tts_engine import synthesize_speech
from src.stt_engine import transcribe_audio

load_dotenv()

# Streamlit Page Config - Light Mode Enforced
st.set_page_config(
    page_title="AgroSageGPT | Digital Agronomist",
    page_icon=":material/psychology:",
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
    
    /* Scrollable chat body with bottom clearance */
    .stMainBlockContainer {
        padding-bottom: 150px !important;
    }
    
    /* Modern Floating Action Dock at downside */
    div[data-testid="stBottom"] {
        background-color: transparent !important;
        padding-bottom: 10px !important;
    }
    div[data-testid="stBottom"] > div {
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(12px);
        border: 1px solid #dce8dc;
        border-radius: 20px;
        padding: 8px 14px;
        box-shadow: 0 4px 20px rgba(0, 40, 0, 0.08);
        max-width: 840px;
        margin: 0 auto;
    }
    div[data-testid="stBottom"] button {
        border-radius: 16px !important;
        font-size: 0.85rem !important;
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

if "current_language" not in st.session_state:
    st.session_state.current_language = "English"

if "mic_key_idx" not in st.session_state:
    st.session_state.mic_key_idx = 0

if "uploaded_doc_names" not in st.session_state:
    st.session_state.uploaded_doc_names = []

if "staged_image" not in st.session_state:
    st.session_state.staged_image = None

if "staged_image_name" not in st.session_state:
    st.session_state.staged_image_name = None

# Sidebar for Preferences & Settings
with st.sidebar:
    st.markdown("### Farm Configuration")
    
    language_mode = st.selectbox(
        "Language Mode:",
        ["Auto-Detect (Adaptive)", "English", "Hindi", "Kannada", "Telugu", "Tamil"],
        index=0,
        help="In Auto-Detect mode, AgroSageGPT dynamically adapts its replies and voice to match the language you speak or type."
    )
    st.caption(f"Active Language: **{st.session_state.current_language}**")
    
    crop_type = st.selectbox(
        "Primary Crop:",
        ["Rice / Paddy", "Wheat", "Tomato", "Maize / Corn", "Cotton", "Chili", "Potato"],
        index=0
    )
    
    city_input = st.text_input("Farm City:", value=os.getenv("DEFAULT_LOCATION", "Bengaluru"))
    if st.button("Sync Weather Location", icon=":material/sync:", use_container_width=True):
        st.session_state.weather_data = get_weather_data(city_input)
        st.session_state.weather_summary = format_weather_summary(st.session_state.weather_data)
        st.toast(f"Synced weather for {city_input}", icon=":material/check:")

    st.markdown("---")
    enable_tts = st.toggle("Read Answers Aloud (TTS)", value=True)
    
    st.markdown("---")
    st.markdown("### Custom Knowledge Base")
    st.caption("Documents dynamically ingested into ChromaDB:")
    if st.session_state.uploaded_doc_names:
        for doc_name in st.session_state.uploaded_doc_names:
            st.markdown(f"• `{doc_name}`")
    else:
        st.caption("*No custom files uploaded yet. Default agronomic KB active.*")
        
    st.markdown("---")
    if st.button("Clear Chat History", icon=":material/delete:", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Chat cleared. What would you like to discuss about your crops today?", "engine": "AgroSage Reset", "audio_bytes": None, "attached_image": None}
        ]
        st.session_state.active_vision_summary = "No crop disease detected. General field management."
        st.rerun()

# ----------------- TOP BRANDING & TELEMETRY BAR -----------------
st.markdown("""
<div class="top-brand">
    <div>
        <div class="brand-title">AgroSageGPT</div>
        <div class="brand-subtitle">Multilingual Generative AI Digital Agronomist • Professional Edition</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Compact Horizontal Telemetry Ribbon
w = st.session_state.weather_data
s = st.session_state.soil_dict

with st.expander(f"Field Telemetry ({w.get('city', 'Farm')}) — {w['temperature_c']}°C | Humidity {w['humidity_pct']}% | NPK {s['nitrogen']['value']}-{s['phosphorus']['value']}-{s['potassium']['value']} | pH {s['ph']['value']}", icon=":material/sensors:", expanded=False):
    t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns(5)
    t_c1.metric("Temperature", f"{w['temperature_c']} °C", f"Feels {w['feels_like_c']} °C")
    t_c2.metric("Humidity", f"{w['humidity_pct']} %", w['condition'])
    t_c3.metric("Wind Speed", f"{w['wind_speed_ms']} m/s", "Drift safe" if w['wind_speed_ms'] < 4.5 else "High drift")
    t_c4.metric("NPK Level", f"{s['nitrogen']['value']}-{s['phosphorus']['value']}-{s['potassium']['value']}", "Optimal" if s['nitrogen']['status'] == "Medium" else "Review")
    t_c5.metric("Soil Moisture", f"{s['moisture']['value']} %", s['moisture']['status'])
    
    st.caption(f"Weather Source: {w.get('source')} | Soil: 7-in-1 IoT Telemetry")
    if st.button("Simulate New Soil Telemetry Event", icon=":material/refresh:", type="secondary"):
        st.session_state.soil_df = generate_soil_history(hours=24)
        st.session_state.soil_dict = get_latest_soil_reading(st.session_state.soil_df, as_dict=True)
        st.session_state.soil_summary = get_latest_soil_reading(st.session_state.soil_df, as_dict=False)
        st.rerun()

# ----------------- CHAT CONVERSATION FEED -----------------
chat_container = st.container()

with chat_container:
    for idx, msg in enumerate(st.session_state.chat_history):
        avatar = ":material/smart_toy:" if msg["role"] == "assistant" else ":material/person:"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("attached_image") is not None:
                st.image(msg["attached_image"], caption="Attached Crop Foliage", width=300)
            
            st.markdown(msg["content"])
            
            if msg.get("audio_bytes"):
                st.audio(msg["audio_bytes"], format="audio/mp3")
                
            if msg.get("engine"):
                st.markdown(f"<span class='engine-tag'>Engine: {msg['engine']}</span>", unsafe_allow_html=True)

# ----------------- FLOATING BOTTOM ACTION PILL & INPUT DOCK -----------------
trigger_diagnose_now = False
voice_text_query = None
quick_chip_query = None

with st.bottom:
    # 1. Staged Leaf Image Banner (if user selected/uploaded an image)
    if st.session_state.staged_image is not None:
        with st.container():
            st_col1, st_col2, st_col3 = st.columns([1, 4, 2])
            st_col1.image(st.session_state.staged_image, width=50)
            st_col2.caption(f"**{st.session_state.staged_image_name or 'Leaf Photo Attached'}** (Ready to analyze with your question)")
            if st_col3.button("Remove", icon=":material/close:", key="clear_staged_leaf", use_container_width=True):
                st.session_state.staged_image = None
                st.session_state.staged_image_name = None
                st.rerun()

    # 2. Sleek Floating Tool Strip: Compact popovers in a single horizontal pill bar
    tool_col1, tool_col2, tool_col3, tool_col4 = st.columns(4)

    # Tool 1: Leaf Photo Popover
    with tool_col1:
        with st.popover("Leaf Photo", icon=":material/photo_camera:", use_container_width=True, help="Upload foliage photo for dual-tier disease diagnosis"):
            st.markdown("##### Upload Foliage Photo")
            st.caption("Local Meta DINOv2 ViT + Gemini Multimodal Validator")
            up_img = st.file_uploader("Choose leaf image", type=["jpg", "jpeg", "png"], key="dock_leaf_up")
            if up_img is not None:
                st.session_state.staged_image = Image.open(up_img)
                st.session_state.staged_image_name = up_img.name
                st.success(f"Attached `{up_img.name}`")

            st.write("Or choose standard samples:")
            s_c1, s_c2 = st.columns(2)
            if s_c1.button("Diseased Leaf Sample", icon=":material/science:", use_container_width=True):
                if os.path.exists("sample_data/sample_diseased_leaf.jpg"):
                    st.session_state.staged_image = Image.open("sample_data/sample_diseased_leaf.jpg")
                    st.session_state.staged_image_name = "sample_diseased_leaf.jpg"
                    st.rerun()
            if s_c2.button("Healthy Leaf Sample", icon=":material/check_circle:", use_container_width=True):
                if os.path.exists("sample_data/sample_healthy_leaf.jpg"):
                    st.session_state.staged_image = Image.open("sample_data/sample_healthy_leaf.jpg")
                    st.session_state.staged_image_name = "sample_healthy_leaf.jpg"
                    st.rerun()

            if st.session_state.staged_image is not None:
                st.image(st.session_state.staged_image, width=160, caption="Staged Foliage")
                if st.button("Diagnose Foliage Now", icon=":material/biotech:", type="primary", use_container_width=True):
                    trigger_diagnose_now = True

    # Tool 2: Voice Microphone Popover
    with tool_col2:
        with st.popover("Voice Mic", icon=":material/mic:", use_container_width=True, help="Speak your question in English, Hindi, Kannada, Telugu, Tamil"):
            v_h1, v_h2 = st.columns([3, 1])
            with v_h1:
                st.markdown(f"##### Voice Query ({st.session_state.current_language})")
            with v_h2:
                if st.button("Reset", icon=":material/refresh:", help="Clear & record again"):
                    st.session_state.mic_key_idx += 1
                    st.rerun()
            st.caption("Noise-gated field audio with Gemini Flash & Whisper ASR")
            mic_audio = st.audio_input(
                "Record voice question",
                key=f"float_dock_mic_{st.session_state.mic_key_idx}"
            )
            if mic_audio is not None:
                with st.spinner("Transcribing speech & adapting language..."):
                    t_text, _ = transcribe_audio(mic_audio, language="Auto")
                    if t_text and len(t_text.strip()) > 3:
                        voice_text_query = t_text
                        if language_mode == "Auto-Detect (Adaptive)":
                            st.session_state.current_language = detect_language(voice_text_query, fallback=st.session_state.current_language)
                        st.info(f"**Recognized ({st.session_state.current_language}):** *\"{voice_text_query}\"*")
                        if st.button("Ask Recognized Question", icon=":material/send:", type="primary", use_container_width=True):
                            pass

    # Tool 3: Dynamic PDF / Manual Ingestion Popover
    with tool_col3:
        with st.popover("Add Guide", icon=":material/upload_file:", use_container_width=True, help="Upload agricultural bulletins or seed manuals into ChromaDB"):
            st.markdown("##### Ingest Agricultural Guides")
            st.caption("Uploaded documents are indexed into ChromaDB vector memory in real-time.")
            up_doc = st.file_uploader("Upload .pdf, .txt, .md", type=["pdf", "txt", "md"], key="float_doc_up")
            if up_doc is not None and up_doc.name not in st.session_state.uploaded_doc_names:
                with st.spinner(f"Indexing `{up_doc.name}` into ChromaDB..."):
                    doc_bytes = up_doc.read()
                    kb = get_knowledge_store()
                    num_chunks = kb.ingest_user_document(doc_bytes, up_doc.name)
                    st.session_state.uploaded_doc_names.append(up_doc.name)
                    st.toast(f"Indexed {num_chunks} chunks from `{up_doc.name}`", icon=":material/check:")
            if st.session_state.uploaded_doc_names:
                st.caption("Currently active manuals:")
                for d in st.session_state.uploaded_doc_names:
                    st.caption(f"• `{d}`")

    # Tool 4: Quick Suggestion Prompts Popover
    with tool_col4:
        with st.popover("Prompts", icon=":material/lightbulb:", use_container_width=True, help="Fast agricultural starter questions"):
            st.markdown("##### Suggested Queries")
            if st.button("When & how to cultivate Rice / Paddy?", icon=":material/help_outline:", use_container_width=True):
                quick_chip_query = "When and how to cultivate Rice (Paddy)? Please provide a complete guide from nursery preparation to harvest."
            if st.button("Does my soil need fertilizer?", icon=":material/help_outline:", use_container_width=True):
                quick_chip_query = "Based on my live soil NPK, pH, and moisture levels, what fertilizers should I apply?"
            if st.button("Is today safe for pesticide spray?", icon=":material/help_outline:", use_container_width=True):
                quick_chip_query = "Considering current temperature, humidity, and wind conditions, is it safe to spray pesticides today?"

    # 3. Main Chat Input Box (floating alongside tool icons)
    user_text_input = st.chat_input("Ask anything about your crops, soil, weather, or attach foliage/docs...")

# ----------------- EXECUTION CONTROLLER -----------------

# Case A: User triggered direct foliage diagnosis button in Leaf Popover
if trigger_diagnose_now and st.session_state.staged_image is not None:
    staged_img = st.session_state.staged_image
    st.session_state.staged_image = None
    st.session_state.staged_image_name = None
    
    with st.spinner("Diagnosing foliage with Hugging Face Meta DINOv2 & Gemini Validator..."):
        v_res = analyze_crop_image(staged_img, crop_type=crop_type)
        diag = v_res["disease_diagnosis"]
        st.session_state.active_vision_summary = v_res["summary"]
        
        val_telemetry = ""
        if "hf_prediction" in diag:
            val_telemetry += f"- **Local Model (HF DINOv2):** `{diag['hf_prediction']}`\n"
        if "validator_verdict" in diag and diag["validator_verdict"]:
            val_telemetry += f"- **Gemini Multimodal Validator:** `{diag['validator_verdict']}` — *{diag.get('validation_notes', '')}*\n"
        
        diag_msg = (
            f"### Plant Pathology Assessment: **{diag['condition']}**\n\n"
            f"{val_telemetry}"
            f"- **Severity:** `{diag['severity']}` (Confidence: {diag['confidence']*100:.1f}%)\n"
            f"- **Observed Symptoms:** {diag['symptoms']}\n"
            f"- **Recommended Field Action:** {diag['recommendation']}\n\n"
            f"*Engine Pipeline: {diag.get('source', 'Vision Engine')}*"
        )
        
        audio_bytes = None
        if enable_tts:
            audio_bytes, _ = synthesize_speech(f"{diag['condition']}. {diag['recommendation']}", language=st.session_state.current_language)
            
        st.session_state.chat_history.append({
            "role": "user",
            "content": f"Please diagnose this {crop_type} crop foliage photo.",
            "attached_image": staged_img,
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

# Case B: Standard Query Resolution (Typed Text > Voice Query > Quick Suggestion Chips)
final_query = user_text_input or voice_text_query or quick_chip_query

if final_query:
    last_user_msg = None
    for m in reversed(st.session_state.chat_history):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break
            
    if final_query != last_user_msg:
        # Dynamic Language Adaptation
        if language_mode == "Auto-Detect (Adaptive)":
            active_language = detect_language(final_query, fallback=st.session_state.current_language)
            st.session_state.current_language = active_language
            lang_label = f"{active_language} (Auto-Detected)"
        else:
            active_language = language_mode
            st.session_state.current_language = active_language
            lang_label = active_language

        # If a leaf image was attached, analyze it to enrich context
        attached_img = None
        if st.session_state.staged_image is not None:
            attached_img = st.session_state.staged_image
            st.session_state.staged_image = None
            st.session_state.staged_image_name = None
            try:
                v_res = analyze_crop_image(attached_img, crop_type=crop_type)
                st.session_state.active_vision_summary = v_res["summary"]
            except Exception as e:
                print(f"[VisionApp] Image analysis inline warning: {e}")

        # Append User Message to Chat History
        st.session_state.chat_history.append({
            "role": "user",
            "content": final_query,
            "attached_image": attached_img,
            "audio_bytes": None,
            "engine": None
        })
        
        # Prepare context inputs
        v_summary = st.session_state.active_vision_summary
        s_summary = st.session_state.soil_summary
        w_summary = st.session_state.weather_summary
        
        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            with st.spinner(f"AgroSageGPT is synthesizing your agronomic guidance ({active_language})..."):
                response_data = ask_agronomist(
                    query=final_query,
                    vision_summary=v_summary,
                    soil_summary=s_summary,
                    weather_summary=w_summary,
                    language=active_language,
                    chat_history=st.session_state.chat_history
                )
                answer_text = response_data["answer"]
                engine_used = response_data.get("engine", "AgroSage Engine")
                engine_display = f"{engine_used} • Language: {lang_label}"
                
                st.markdown(answer_text)
                
                # Synthesize High-Definition Neural Audio
                audio_bytes = None
                if enable_tts:
                    with st.spinner(f"Synthesizing neural voice in {active_language}..."):
                        audio_bytes, tts_engine_name = synthesize_speech(answer_text, language=active_language)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            st.caption(f"Audio playback: {tts_engine_name} ({active_language})")
                            
                st.markdown(f"<span class='engine-tag'>Engine: {engine_display}</span>", unsafe_allow_html=True)
                
                with st.expander("Retrieved Agronomic Knowledge Chunks", icon=":material/menu_book:", expanded=False):
                    for idx, doc in enumerate(response_data.get("retrieved_context", [])):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.text(doc)
                        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
            "attached_image": None,
            "engine": engine_display,
            "audio_bytes": audio_bytes
        })
        st.rerun()
