from __future__ import annotations

import asyncio as _asyncio
import queue
import re
import threading
from collections.abc import Iterator

import edge_tts

from audio import play_audio_bytes
from config import TTS_VOICE, TTS_RATE

# Single persistent event loop in a background thread — eliminates per-call
# event loop creation/destruction overhead (~50-100ms per sentence saved).
_loop = _asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="tts-loop").start()


def _preprocess_for_tts(text: str) -> str:
    """Expand abbreviations that TTS engines commonly mispronounce."""
    # Titles / honorifics — before a word (usually a name)
    text = re.sub(r'\bMr\.(?=\s)', 'Mister', text)
    text = re.sub(r'\bMrs\.(?=\s)', 'Missus', text)
    text = re.sub(r'\bMs\.(?=\s)', 'Miss', text)
    text = re.sub(r'\bDr\.(?=\s[A-Z])', 'Doctor', text)
    text = re.sub(r'\bProf\.(?=\s)', 'Professor', text)
    text = re.sub(r'\bJr\.', 'Junior', text)
    text = re.sub(r'\bSr\.(?=\s|$)', 'Senior', text)

    # St. — Saint before a capitalised name, Street otherwise
    text = re.sub(r'\bSt\.(?=\s+[A-Z])', 'Saint', text)
    text = re.sub(r'\bSt\.', 'Street', text)

    # Road / address suffixes
    text = re.sub(r'\bAve\.', 'Avenue', text)
    text = re.sub(r'\bBlvd\.', 'Boulevard', text)
    text = re.sub(r'\bRd\.', 'Road', text)
    text = re.sub(r'\bLn\.', 'Lane', text)
    text = re.sub(r'\bDr\.(?=\s|,|$)', 'Drive', text)   # remaining Dr. = Drive

    # Latin / common shorthand
    text = re.sub(r'\betc\.', 'etcetera', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.,?', 'for example,', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\.e\.,?', 'that is,', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvs\.', 'versus', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNo\.(?=\s*\d)', 'Number', text, flags=re.IGNORECASE)

    return text


def _run_async(coro):
    """Submit a coroutine to the persistent loop and block until it completes."""
    return _asyncio.run_coroutine_threadsafe(coro, _loop).result()


# Match a sentence boundary: punctuation followed by whitespace (or end of string)
_SENTENCE_END = re.compile(r'(?<=[.!?])(\s+|$)')

# Fixed phrases pre-synthesised at startup so speak() serves them from memory
# with no network round-trip (~400ms saved per call).
_WARM_PHRASES = [
    "Yes?",
    "I didn't catch that.",
    "Goodbye!",
    "Happy to help! Goodbye.",
    "Done, memory cleared. What else?",
]


class Speaker:
    """Converts text to speech via edge-tts and plays it back."""

    _cache: dict[str, bytes] = {}

    def __init__(self, voice: str = TTS_VOICE, rate: str = TTS_RATE):
        self.voice = voice
        self.rate = rate
        self._prewarm()

    def _prewarm(self) -> None:
        print("[Sam] Pre-synthesising phrases...")
        for phrase in _WARM_PHRASES:
            mp3 = _run_async(self._synthesise(phrase))
            if mp3:
                Speaker._cache[phrase] = mp3
        print("[Sam] Ready.")

    def speak(self, text: str) -> None:
        """Play text — serves from cache for known phrases, synthesises otherwise."""
        text = text.strip()
        if not text:
            return
        mp3 = Speaker._cache.get(text) or _run_async(self._synthesise(text))
        if mp3:
            play_audio_bytes(mp3)

    def speak_streaming(self, token_iter: Iterator[str]) -> str:
        """
        Consume a Claude token stream, synthesise each sentence as it completes,
        and play sentences in a background thread so synthesis and playback overlap.

        Timeline:
          Claude streams tokens → sentence buffer fills → first sentence synthesised
          → playback thread starts playing sentence 1
          → main thread synthesises sentence 2 while sentence 1 plays
          → ...

        Returns the full response text.
        """
        audio_q: queue.Queue[bytes | None] = queue.Queue()

        def _playback_worker() -> None:
            while True:
                item = audio_q.get()
                if item is None:
                    break
                play_audio_bytes(item)

        playback_thread = threading.Thread(target=_playback_worker, daemon=True)
        playback_thread.start()

        full_response = ""
        buffer = ""

        for token in token_iter:
            full_response += token
            buffer += token
            # Flush every complete sentence to the playback queue immediately
            while True:
                m = _SENTENCE_END.search(buffer)
                if not m:
                    break
                sentence = buffer[: m.start()].strip()
                buffer = buffer[m.end() :]
                if sentence:
                    mp3 = _run_async(self._synthesise(sentence))
                    if mp3:
                        audio_q.put(mp3)

        # Synthesise any remaining text that didn't end with punctuation
        if buffer.strip():
            mp3 = _run_async(self._synthesise(buffer.strip()))
            if mp3:
                audio_q.put(mp3)

        audio_q.put(None)          # signal playback thread to exit
        playback_thread.join()     # wait until last sentence finishes playing
        return full_response

    async def _synthesise(self, text: str) -> bytes:
        text = _preprocess_for_tts(text)
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)
