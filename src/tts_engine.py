"""Text-to-Speech (TTS) Engine for AgroSageGPT.

Provides spoken audio advisories for farmers in regional languages
(English, Hindi, Kannada, Telugu, Tamil).
Supports Hugging Face MMS (facebook/mms-tts) and high-speed streaming multilingual TTS.
"""

import os
import io
import re
from typing import Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

# Language code mappings for TTS engines
LANG_CODE_MAP = {
    "english": {"gtts": "en", "hf_mms": "facebook/mms-tts-eng"},
    "hindi": {"gtts": "hi", "hf_mms": "facebook/mms-tts-hin"},
    "kannada": {"gtts": "kn", "hf_mms": "facebook/mms-tts-kan"},
    "telugu": {"gtts": "te", "hf_mms": "facebook/mms-tts-tel"},
    "tamil": {"gtts": "ta", "hf_mms": "facebook/mms-tts-tam"}
}


def clean_text_for_speech(text: str) -> str:
    """Pre-processes LLM markdown output to ensure natural sounding speech.
    
    Removes markdown headers (#), asterisks (**), emojis, bullet symbols,
    and metadata lines so the voice reads like a human agronomist.
    """
    if not text:
        return ""
    
    # Remove markdown code fences and backticks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    
    # Remove markdown headings (e.g. ###, ##, #)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    # Remove bold/italic asterisks & underscores
    text = re.sub(r"[\*_]{1,3}([^\*_]+)[\*_]{1,3}", r"\1", text)
    
    # Remove emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    
    # Remove metadata lines or bullet markers
    text = re.sub(r"^[\s\-\*•\d\.]+\s*", "", text, flags=re.MULTILINE)
    
    # Normalize multiple whitespace and newlines into natural pauses
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def synthesize_speech_hf_mms(text: str, language: str = "Hindi") -> Optional[bytes]:
    """Synthesize speech using Hugging Face MMS Inference API."""
    hf_token = os.getenv("HF_TOKEN", "").strip()
    lang_info = LANG_CODE_MAP.get(language.lower(), LANG_CODE_MAP["english"])
    model_id = lang_info["hf_mms"]
    
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text[:400]}, timeout=8)
        if response.status_code == 200 and len(response.content) > 100:
            return response.content
    except Exception as e:
        print(f"[TTSEngine] Hugging Face inference note: {e}")
        
    return None


def synthesize_speech(text: str, language: str = "English") -> Tuple[Optional[bytes], str]:
    """Convert text advisory into spoken audio bytes.
    
    Args:
        text: Markdown or plain text from agronomist RAG.
        language: Target regional language (English, Hindi, Kannada, Telugu, Tamil).
        
    Returns:
        tuple: (audio_bytes or None, engine_used: str)
    """
    clean_speech_text = clean_text_for_speech(text)
    if not clean_speech_text:
        return None, "Empty text"
    
    # Shorten for speech clarity if excessively long
    if len(clean_speech_text) > 1200:
        clean_speech_text = clean_speech_text[:1200] + "..."

    lang_lower = language.lower()
    lang_info = LANG_CODE_MAP.get(lang_lower, LANG_CODE_MAP["english"])

    # 1. Try Hugging Face MMS if HF_TOKEN is configured
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token and hf_token != "your_hf_token_here":
        hf_audio = synthesize_speech_hf_mms(clean_speech_text, language=language)
        if hf_audio:
            return hf_audio, f"Hugging Face Meta MMS ({lang_info['hf_mms']})"

    # 2. Fast Streaming Multilingual Engine (gTTS)
    try:
        from gtts import gTTS
        gtts_code = lang_info["gtts"]
        tts = gTTS(text=clean_speech_text, lang=gtts_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read(), f"Multilingual Neural TTS ({language})"
    except Exception as e:
        print(f"[TTSEngine] gTTS error: {e}")
        return None, f"TTS Error: {str(e)}"
