SAMPLE_RATE: int = 16000
CHANNELS: int = 1
FRAME_DURATION_MS: int = 30
FRAME_SAMPLES: int = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480

VAD_AGGRESSIVENESS: int = 2       # 0-3; 1 is less aggressive so quiet speech at word edges isn't dropped
SILENCE_GAP_MS: int = 400        # flush wake-word segment after this much silence
COMMAND_SILENCE_MS: int = 500   # end command recording after this much silence (was 750 — too short for natural pauses)
COMMAND_MAX_DURATION_S: int = 15
PRE_SPEECH_BUFFER_FRAMES: int = 5  # ~150ms ring buffer prepended so word onsets aren't clipped

WAKE_MODEL: str = "tiny"
COMMAND_MODEL: str = "base.en"  # English-only base: same size as base, faster + slightly better accuracy than multilingual base
COMPUTE_TYPE: str = "int8"

CLAUDE_MODEL: str = "claude-haiku-4-5-20251001"
MAX_HISTORY_TURNS: int = 20

TTS_VOICE: str = "en-US-AvaNeural"
TTS_RATE: str = "+20%"

CONVERSATION_TIMEOUT_S: float = 6.0   # seconds of silence before returning to wake-word mode

INPUT_DEVICE: int | None = None
OUTPUT_DEVICE: int | None = None

SYSTEM_PROMPT: str = (
    "You are Sam, a helpful voice assistant running on the user's local machine. "
    "Keep your responses concise and conversational — you are speaking aloud, not writing. "
    "Avoid bullet points, markdown formatting, long lists, and emojis under any circumstances. "
    "Speak naturally as if having a conversation. "
    "\n\n"
    "You have access to the following live integrations that are automatically triggered "
    "based on what the user says:\n"
    "- Weather: current conditions and 4-day forecast for the user's location\n"
    "- News: latest headlines from Currents API\n"
    "- Google Calendar: upcoming events from the user's calendar\n"
    "- Spotify: play songs/artists/playlists, skip, pause, volume, shuffle, repeat, like songs\n"
    "- Web Search: DuckDuckGo search results for factual queries\n"
    "- Wikipedia: article summaries for knowledge questions\n"
    "\n"
    "When you receive a [System: ...] prefix in a message, it contains live data fetched "
    "from one of these integrations. Use it to answer the user's question naturally — "
    "do not mention the system prefix, the data source name, or that data was fetched. "
    "Just answer as if you know the information. "
    "If the system data contains an error, apologise briefly and suggest the user try again."
)
