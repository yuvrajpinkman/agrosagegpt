"""Text-to-Speech (TTS) Engine for AgroSageGPT.

Provides studio-quality neural voice readouts for farmers in regional languages
(Hindi, Kannada, Telugu, Tamil, and English).
Primary: High-Definition Neural Regional Voices (edge-tts)
Fallback 1: Hugging Face Meta MMS (facebook/mms-tts)
Fallback 2: Google Multilingual Neural TTS (gTTS)
"""

import os
import io
import re
import asyncio
from typing import Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

# Neural Regional Voices (Authentic native accents & cadence)
NEURAL_VOICE_MAP = {
    "hindi": "hi-IN-MadhurNeural",
    "kannada": "kn-IN-GaganNeural",
    "telugu": "te-IN-MohanNeural",
    "tamil": "ta-IN-PallaviNeural",
    "english": "en-IN-NeerjaNeural"
}

# Hugging Face MMS model IDs
HF_MMS_MAP = {
    "hindi": "facebook/mms-tts-hin",
    "kannada": "facebook/mms-tts-kan",
    "telugu": "facebook/mms-tts-tel",
    "tamil": "facebook/mms-tts-tam",
    "english": "facebook/mms-tts-eng"
}

GTTS_CODE_MAP = {
    "hindi": "hi",
    "kannada": "kn",
    "telugu": "te",
    "tamil": "ta",
    "english": "en"
}


def clean_text_for_speech(text: str) -> str:
    """Pre-processes LLM markdown output to ensure natural sounding speech.
    
    Removes markdown headers (#), asterisks (**), emojis, bullet symbols,
    and metadata lines so the voice reads like a human agronomist.
    """
    if not text:
        return ""
    
    # Remove code blocks and backticks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    
    # Remove markdown headings (e.g. ###, ##, #)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    
    # Remove bold/italic asterisks & underscores
    text = re.sub(r"[\*_]{1,3}([^\*_]+)[\*_]{1,3}", r"\1", text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    
    # Remove bullet markers and list numbers
    text = re.sub(r"^[\s\-\*•\d\.]+\s*", "", text, flags=re.MULTILINE)
    
    # Normalize pauses
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


async def _synthesize_edge_tts(text: str, voice_name: str) -> Optional[bytes]:
    """Synthesize high-fidelity audio using edge-tts async stream."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice_name)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data if len(audio_data) > 200 else None


def synthesize_speech(text: str, language: str = "English") -> Tuple[Optional[bytes], str]:
    """Convert text advisory into natural spoken audio bytes.
    
    Args:
        text: Markdown or plain text from agronomist RAG.
        language: Target regional language (English, Hindi, Kannada, Telugu, Tamil).
        
    Returns:
        tuple: (audio_bytes or None, engine_used: str)
    """
    clean_speech_text = clean_text_for_speech(text)
    if not clean_speech_text:
        return None, "Empty text"
    
    # Crop to ~1000 characters for optimal conversational pace
    if len(clean_speech_text) > 1000:
        clean_speech_text = clean_speech_text[:1000] + "..."

    lang_lower = language.lower()
    voice_name = NEURAL_VOICE_MAP.get(lang_lower, NEURAL_VOICE_MAP["english"])

    # 1. PRIMARY: High-Definition Neural Voice (edge-tts)
    try:
        audio_bytes = asyncio.run(_synthesize_edge_tts(clean_speech_text, voice_name))
        if audio_bytes:
            return audio_bytes, f"High-Definition Neural Voice ({voice_name})"
    except Exception as e:
        print(f"[TTSEngine] Primary Neural TTS error: {e}. Trying fallback...")

    # 2. SECONDARY: Hugging Face Meta MMS
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token and hf_token != "your_hf_token_here":
        mms_model = HF_MMS_MAP.get(lang_lower, HF_MMS_MAP["english"])
        api_url = f"https://api-inference.huggingface.co/models/{mms_model}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        try:
            res = requests.post(api_url, headers=headers, json={"inputs": clean_speech_text[:350]}, timeout=8)
            if res.status_code == 200 and len(res.content) > 200:
                return res.content, f"Hugging Face Meta MMS ({mms_model})"
        except Exception as e:
            print(f"[TTSEngine] Hugging Face MMS error: {e}")

    # 3. TERTIARY: Google Neural TTS (gTTS)
    try:
        from gtts import gTTS
        gtts_code = GTTS_CODE_MAP.get(lang_lower, "en")
        tts = gTTS(text=clean_speech_text, lang=gtts_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read(), f"Google Neural TTS ({language})"
    except Exception as e:
        print(f"[TTSEngine] gTTS error: {e}")

    return None, "TTS Error: Audio synthesis failed"
