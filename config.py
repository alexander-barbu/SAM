import os

SAMPLE_RATE: int = 16000
CHANNELS: int = 1
FRAME_DURATION_MS: int = 30
FRAME_SAMPLES: int = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480

VAD_AGGRESSIVENESS: int = 2
SILENCE_GAP_MS: int = 400        # flush wake-word segment after this much silence
COMMAND_SILENCE_MS: int = 750   # end command recording after this much silence
COMMAND_MAX_DURATION_S: int = 15

WAKE_MODEL: str = "tiny"
COMMAND_MODEL: str = "base"
COMPUTE_TYPE: str = "int8"

CLAUDE_MODEL: str = "claude-haiku-4-5-20251001"
MAX_HISTORY_TURNS: int = 20

TTS_VOICE: str = "en-US-AvaNeural"
TTS_RATE: str = "+20%"

CONVERSATION_TIMEOUT_S: float = 6.0   # seconds of silence before returning to wake-word mode

INPUT_DEVICE: int | None = None
OUTPUT_DEVICE: int | None = None

SYSTEM_PROMPT: str = (
    "You are Sam, a helpful voice assistant. "
    "Keep your responses concise and conversational — you are speaking aloud, not writing. "
    "Avoid bullet points, markdown formatting, long lists, and emojis. "
    "Never use emojis under any circumstances. "
    "Speak naturally as if having a conversation."
)
