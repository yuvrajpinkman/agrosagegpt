"""Speech-to-Text (STT) Module for AgroSageGPT.

Wraps the complete agricultural audio pipeline:
- Gated noise filtering (tractor / pump suppression)
- Primary ASR: Google Gemini Flash Multimodal Audio ASR
- Failover ASR: Groq Whisper-Large-v3-Turbo
- Domain-specific word restoration (Agri Lexicon + phonetic matching)
"""

import io
from typing import Optional, Tuple, Union
from src.audio_pipeline import process_noisy_field_audio


def transcribe_audio(
    audio_data: Union[bytes, io.BytesIO, str],
    language: str = "Auto",
    filename: str = "voice_input.wav"
) -> Tuple[Optional[str], str]:
    """Transcribes and restores agricultural field voice query.
    
    Args:
        audio_data: Audio bytes, BytesIO buffer (from st.audio_input), or file path.
        language: Selected regional language preference for better decoding accuracy.
        filename: Optional filename hint.
        
    Returns:
        tuple: (corrected_query_text or None, engine_used_summary: str)
    """
    corrected_text, meta = process_noisy_field_audio(audio_data, language=language)
    
    if corrected_text:
        asr_engine = meta.get("asr_engine", "Model-Agnostic ASR")
        filter_status = meta.get("filtering_applied", "Noise Gating")
        engine_summary = f"{asr_engine} + Agri Lexicon Restoration ({filter_status})"
        return corrected_text, engine_summary
    else:
        err = meta.get("error", "Unable to transcribe field audio")
        return None, f"STT Error: {err}"
