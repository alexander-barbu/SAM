import queue
import numpy as np
import sounddevice as sd
import miniaudio

import webrtcvad

from config import (
    SAMPLE_RATE, CHANNELS, FRAME_DURATION_MS, FRAME_SAMPLES,
    VAD_AGGRESSIVENESS, COMMAND_SILENCE_MS, COMMAND_MAX_DURATION_S,
    PRE_SPEECH_BUFFER_FRAMES, OUTPUT_DEVICE, MIN_SPEECH_RMS,
)


class AudioRecorder:
    """Continuous mic stream with VAD gating."""

    def __init__(self, device: int | None = None, vad_aggressiveness: int = VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.device = device
        self._q: queue.Queue[bytes] = queue.Queue()
        self._stream: sd.RawInputStream | None = None

    def open(self) -> None:
        def _callback(indata, frames, time, status):
            self._q.put(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=_callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self) -> bytes:
        """Block until one 30ms PCM frame is available."""
        return self._q.get()

    def drain(self) -> None:
        """Discard all frames that accumulated while Sam was speaking."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def is_speech(self, frame: bytes) -> bool:
        pcm = np.frombuffer(frame, dtype=np.int16)
        rms = np.sqrt(np.mean(pcm.astype(np.float32) ** 2))
        if rms < MIN_SPEECH_RMS:
            return False
        return self.vad.is_speech(frame, SAMPLE_RATE)

    def record_until_silence(
        self,
        silence_ms: int = COMMAND_SILENCE_MS,
        max_duration_s: int = COMMAND_MAX_DURATION_S,
        speech_timeout_s: float | None = None,
    ) -> np.ndarray:
        """
        Record from mic, stopping when silence_ms of consecutive silence
        is detected after speech begins, or max_duration_s is reached.

        If speech_timeout_s is set and no speech begins within that many
        seconds, returns an empty array (caller treats this as a timeout).

        Returns float32 numpy array suitable for Whisper.
        """
        silence_frames_needed = silence_ms // FRAME_DURATION_MS
        max_frames = max_duration_s * 1000 // FRAME_DURATION_MS
        timeout_frames = int(speech_timeout_s * 1000 / FRAME_DURATION_MS) if speech_timeout_s else None

        voiced_started = False
        consecutive_silence = 0
        pre_speech_count = 0
        # Ring buffer so word onsets aren't clipped when VAD fires slightly late
        pre_speech_ring: list[bytes] = []
        frames: list[bytes] = []
        total_frames = 0

        while total_frames < max_frames:
            frame = self.read_frame()
            total_frames += 1
            speech = self.is_speech(frame)

            if speech:
                if not voiced_started:
                    # Prepend the recent pre-speech frames so the onset isn't lost
                    frames.extend(pre_speech_ring)
                    pre_speech_ring.clear()
                voiced_started = True
                consecutive_silence = 0
                frames.append(frame)
            elif voiced_started:
                consecutive_silence += 1
                frames.append(frame)
                if consecutive_silence >= silence_frames_needed:
                    break
            else:
                # Keep a short rolling window before speech starts
                pre_speech_ring.append(frame)
                if len(pre_speech_ring) > PRE_SPEECH_BUFFER_FRAMES:
                    pre_speech_ring.pop(0)
                pre_speech_count += 1
                if timeout_frames and pre_speech_count >= timeout_frames:
                    return pcm_frames_to_float32([])  # timed out waiting for speech

        return pcm_frames_to_float32(frames)


def pcm_frames_to_float32(frames: list[bytes]) -> np.ndarray:
    """Join raw int16 PCM frames and normalise to float32 in [-1, 1]."""
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def play_audio_bytes(mp3_bytes: bytes, device: int | None = OUTPUT_DEVICE) -> None:
    """Decode MP3 bytes from edge-tts and play synchronously."""
    decoded = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.SIGNED16)
    audio = np.frombuffer(decoded.samples, dtype=np.int16)
    if decoded.nchannels > 1:
        audio = audio.reshape(-1, decoded.nchannels)
    sd.play(audio, samplerate=decoded.sample_rate, device=device)
    sd.wait()


def beep(frequency_hz: int = 880, duration_ms: int = 150, device: int | None = OUTPUT_DEVICE) -> None:
    """Play a short sine-wave acknowledgement tone."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = (np.sin(2 * np.pi * frequency_hz * t) * 0.25).astype(np.float32)
    sd.play(wave, samplerate=SAMPLE_RATE, device=device)
    sd.wait()
