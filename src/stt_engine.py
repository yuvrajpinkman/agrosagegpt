"""Speech-to-Text (STT) Module for AgroSageGPT.

Transcribes spoken farmer voice questions in regional languages
(Hindi, Kannada, Telugu, Tamil, Marathi, Punjabi, Bengali, English).
Powered by Whisper-Large-v3-Turbo via Groq and Hugging Face.
"""

import os
import io
from typing import Optional, Tuple, Union
from dotenv import load_dotenv

load_dotenv()

# Language code mapping for Whisper STT
WHISPER_LANG_MAP = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
    "telugu": "te",
    "tamil": "ta",
    "marathi": "mr",
    "punjabi": "pa",
    "bengali": "bn",
    "gujarati": "gu"
}


def transcribe_audio(
    audio_data: Union[bytes, io.BytesIO, str],
    language: str = "Hindi",
    filename: str = "voice_input.wav"
) -> Tuple[Optional[str], str]:
    """Transcribe spoken voice audio into text.
    
    Args:
        audio_data: Audio bytes, BytesIO buffer (from st.audio_input), or file path.
        language: Selected regional language preference for better decoding accuracy.
        filename: Optional filename hint.
        
    Returns:
        tuple: (transcribed_text or None, engine_used: str)
    """
    # Normalize audio data into bytes
    if isinstance(audio_data, str):
        if not os.path.exists(audio_data):
            return None, f"File not found: {audio_data}"
        with open(audio_data, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(audio_data, io.BytesIO):
        raw_bytes = audio_data.getvalue()
    elif isinstance(audio_data, bytes):
        raw_bytes = audio_data
    else:
        # File-like object (e.g. UploadedFile from Streamlit)
        if hasattr(audio_data, "read"):
            raw_bytes = audio_data.read()
            if hasattr(audio_data, "seek"):
                audio_data.seek(0)
        else:
            return None, "Unsupported audio input type."

    if not raw_bytes or len(raw_bytes) < 100:
        return None, "Audio buffer is empty."

    whisper_lang = WHISPER_LANG_MAP.get(language.lower(), None)

    # 1. PRIMARY: Groq Whisper-Large-v3-Turbo (Ultra-fast, LPU accelerated)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and groq_key != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            
            kwargs = {
                "file": (filename, raw_bytes),
                "model": "whisper-large-v3-turbo",
                "temperature": 0.0
            }
            if whisper_lang:
                kwargs["language"] = whisper_lang

            transcription = client.audio.transcriptions.create(**kwargs)
            text = transcription.text.strip()
            if text:
                return text, "Groq Whisper-Large-v3-Turbo (Live)"
        except Exception as e:
            print(f"[STTEngine] Groq Whisper error: {e}. Trying Hugging Face...")

    # 2. SECONDARY: Hugging Face Whisper Inference API
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token and hf_token != "your_hf_token_here":
        try:
            import requests
            api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
            headers = {"Authorization": f"Bearer {hf_token}"}
            res = requests.post(api_url, headers=headers, data=raw_bytes, timeout=12)
            if res.status_code == 200:
                result = res.json()
                text = result.get("text", "").strip()
                if text:
                    return text, "Hugging Face Whisper-Large-v3"
        except Exception as e:
            print(f"[STTEngine] Hugging Face Whisper error: {e}")

    return None, "STT Error: Unable to transcribe audio (API key missing or format unreadable)"
