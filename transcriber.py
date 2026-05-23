import numpy as np
from faster_whisper import WhisperModel

from config import COMMAND_MODEL, COMPUTE_TYPE


class Transcriber:
    """Transcribes command audio using faster-whisper (base model)."""

    def __init__(self):
        print(f"[Sam] Loading command model (whisper {COMMAND_MODEL})...")
        self.model = WhisperModel(COMMAND_MODEL, device="cpu", compute_type=COMPUTE_TYPE)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 numpy audio array at 16kHz.
        Returns the full transcript stripped of leading/trailing whitespace,
        or an empty string if nothing was recognised.
        """
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
