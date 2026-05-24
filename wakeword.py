import re

import numpy as np
from faster_whisper import WhisperModel

from audio import AudioRecorder, pcm_frames_to_float32
from config import WAKE_MODEL, COMPUTE_TYPE, SILENCE_GAP_MS, FRAME_DURATION_MS

# Whole-word match only — prevents "same", "psalm", "awesome" from triggering
_WAKE_WORD_RE = re.compile(r'\bsam\b')

# Segments where Whisper has low confidence there's real speech are discarded
_NO_SPEECH_THRESHOLD = 0.8

# Voiced frames required before a segment is worth transcribing (~60 ms)
_MIN_VOICED_FRAMES = 2


class WakeWordDetector:
    """
    Continuously reads mic frames, accumulates voiced segments via WebRTC VAD,
    and transcribes each segment with whisper-tiny.
    Returns control to the caller when "sam" is found in the transcription.
    """

    _MAX_SEGMENT_MS = 2000  # force-flush a segment after 2 seconds of speech

    def __init__(self, recorder: AudioRecorder):
        self.recorder = recorder
        self._silence_frames = SILENCE_GAP_MS // FRAME_DURATION_MS
        self._max_segment_frames = self._MAX_SEGMENT_MS // FRAME_DURATION_MS
        print("[Sam] Loading wake-word model (whisper tiny)...")
        self.model = WhisperModel(WAKE_MODEL, device="cpu", compute_type=COMPUTE_TYPE)

    def listen_for_wake_word(self) -> None:
        """Block until 'sam' is detected in a transcribed segment."""
        while True:
            frames = self._collect_voiced_segment()
            if not frames:
                continue
            text = self._transcribe(frames)
            if _WAKE_WORD_RE.search(text):
                return

    def _collect_voiced_segment(self) -> list[bytes]:
        """
        Accumulate voiced frames until a silence gap or max-segment length.
        Returns the list of PCM frames, or an empty list if the voiced portion
        was too short to be worth transcribing (e.g. a noise burst).
        """
        consecutive_silence = 0
        frames: list[bytes] = []
        voiced_started = False
        voiced_count = 0

        while True:
            frame = self.recorder.read_frame()
            speech = self.recorder.is_speech(frame)

            if speech:
                voiced_started = True
                consecutive_silence = 0
                voiced_count += 1
                frames.append(frame)
                if len(frames) >= self._max_segment_frames:
                    return frames
            elif voiced_started:
                consecutive_silence += 1
                frames.append(frame)
                if consecutive_silence >= self._silence_frames:
                    # Discard very short noise bursts (~< 150 ms of real speech)
                    if voiced_count < _MIN_VOICED_FRAMES:
                        return []
                    return frames
            # pure silence before any speech: keep waiting, nothing to return yet

    def _transcribe(self, frames: list[bytes]) -> str:
        audio = pcm_frames_to_float32(frames)
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,      # higher beam for better single-word accuracy
            vad_filter=False, # VAD already applied upstream
            # no hotwords — biasing toward "Sam" causes hallucinations on noise
        )
        parts = []
        for seg in segments:
            # Skip segments where Whisper isn't confident there's real speech
            if seg.no_speech_prob > _NO_SPEECH_THRESHOLD:
                continue
            parts.append(seg.text)
        return " ".join(parts).lower()
