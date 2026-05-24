import re

import numpy as np
from faster_whisper import WhisperModel

from config import COMMAND_MODEL, COMPUTE_TYPE

# Whisper commonly hallucinates these when fed near-silence or noise
_HALLUCINATIONS = re.compile(
    r"^(thank(s| you)[\.\s!]*|you\.?|bye[\.\s]*|"
    r"thanks for watching[\.\s]*|please subscribe[\.\s]*|"
    r"\[.*?\]|\.+|\s*)$",
    re.IGNORECASE,
)

_INITIAL_PROMPT = (
    "This is a voice command for a personal assistant called Sam. "
    "The user may ask about weather, news, music, calendar, or general questions."
)


class Transcriber:
    """Transcribes command audio using faster-whisper."""

    def __init__(self):
        print(f"[Sam] Loading command model (whisper {COMMAND_MODEL})...")
        self.model = WhisperModel(COMMAND_MODEL, device="cpu", compute_type=COMPUTE_TYPE)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 numpy audio array at 16kHz.
        Returns the full transcript or an empty string if nothing meaningful was recognised.
        """
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=2,              # 2 beats greedy (1) for accuracy with minimal speed cost vs 5
            vad_filter=True,
            initial_prompt=_INITIAL_PROMPT,
            condition_on_previous_text=False,
            temperature=0.0,          # single value avoids retry overhead of a fallback list
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if _HALLUCINATIONS.fullmatch(text):
            return ""
        return text
