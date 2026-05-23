import os
import sys

from dotenv import load_dotenv

from audio import AudioRecorder
from config import INPUT_DEVICE, CONVERSATION_TIMEOUT_S
from llm import ConversationManager
from transcriber import Transcriber
from tts import Speaker
from wakeword import WakeWordDetector

_RESET_PHRASES = {"forget everything", "reset conversation", "start over", "clear history"}
_EXIT_PHRASES  = {"goodbye", "bye", "see you", "that's all", "stop", "go to sleep"}
_SLEEP_PHRASES = {
    "thanks a lot", "thank you so much", "thanks for that", "thanks for your help",
    "cheers", "that's all i need", "that's perfect", "that's great", "brilliant thanks",
    "perfect thanks", "great thanks", "thank you",
}


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Sam] Error: ANTHROPIC_API_KEY not set. Create a .env file with your key.")
        sys.exit(1)

    recorder = AudioRecorder(device=INPUT_DEVICE)
    detector = WakeWordDetector(recorder)
    transcriber = Transcriber()
    conversation = ConversationManager(api_key=api_key)
    speaker = Speaker()

    recorder.open()
    print("[Sam] Ready. Say 'Sam' to activate. Press Ctrl+C to quit.\n")

    try:
        while True:
            # ── IDLE: wait for wake word ──────────────────────────────────────
            detector.listen_for_wake_word()
            print("[Sam] Conversation started. (Say 'goodbye' or stay silent to exit.)")
            speaker.speak("Yes?")


            # ── CONVERSATION LOOP ─────────────────────────────────────────────
            while True:
                recorder.drain()  # discard mic bleed from beep/TTS before listening
                print("[Sam] Listening...")
                audio = recorder.record_until_silence(speech_timeout_s=CONVERSATION_TIMEOUT_S)

                # Silence timeout — no speech within the window
                if audio.size == 0:
                    print("[Sam] No response — going back to sleep.\n")
                    break

                user_text = transcriber.transcribe(audio)
                if not user_text:
                    speaker.speak("I didn't catch that.")
                    recorder.drain()
                    continue
                print(f"[You] {user_text}")

                # Exit conversation — explicit farewell
                if any(phrase in user_text.lower() for phrase in _EXIT_PHRASES):
                    speaker.speak("Goodbye!")
                    break

                # Sleep — gratitude / natural conversation ender
                if any(phrase in user_text.lower() for phrase in _SLEEP_PHRASES):
                    speaker.speak("Happy to help! Goodbye.")
                    break

                # Reset history
                if any(phrase in user_text.lower() for phrase in _RESET_PHRASES):
                    conversation.reset()
                    speaker.speak("Done, memory cleared. What else?")
                    recorder.drain()
                    continue

                # Send to Claude and stream response directly to TTS
                print("[Sam] ", end="", flush=True)
                response = speaker.speak_streaming(conversation.stream_tokens(user_text))
                print(f"{response}\n")
                recorder.drain()  # flush mic bleed before next listen

    except KeyboardInterrupt:
        print("\n[Sam] Goodbye.")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
