import os
import re
import sys

from dotenv import load_dotenv

from audio import AudioRecorder
from apis import (get_location, get_weather, get_news,
                  get_duckduckgo_answer, get_wikipedia_summary,
                  get_calendar_events, get_spotify_client, spotify_control)
from config import INPUT_DEVICE, CONVERSATION_TIMEOUT_S
from llm import ConversationManager
from transcriber import Transcriber
from tts import Speaker
from wakeword import WakeWordDetector

_RESET_PHRASES = {"forget everything", "reset conversation", "start over", "clear history"}
_EXIT_PHRASES  = {"goodbye", "bye", "see you", "that's all", "stop", "go to sleep"}
_SLEEP_PHRASES = {
    "thanks a lot", "thank you so much", "thanks for that", "thanks for your help",
    "sleep", "that's all i need", "that's perfect", "that's great", "brilliant thanks",
    "perfect thanks", "great thanks", "thank you", "nope, that's it"
}
_WEATHER_PHRASES = {
    "weather", "temperature", "raining", "rain", "sunny", "cloudy",
    "forecast", "wind", "hot outside", "cold outside", "snowing",
    "snow", "humid", "humidity", "storm", "feels like",
}
_NEWS_PHRASES = {
    "news", "headlines", "what's happening", "current events",
    "latest", "in the news", "tell me about", "any news",
}
_CALENDAR_PHRASES = {
    "calendar", "schedule", "appointments", "meetings", "what do i have",
    "am i free", "what's on my schedule", "any events", "upcoming events",
    "what's today", "what's tomorrow",
}
_SPOTIFY_KEYWORDS = {
    "pause", "unpause", "resume music", "skip", "next song", "previous song",
    "volume up", "volume down", "louder", "quieter", "shuffle",
    "what's playing", "now playing", "current song", "who sings",
    "like this song", "like the song", "unlike", "recently played",
    "repeat", "stop music", "stop the music",
    "add to playlist", "save to playlist", "save this song",
}


def _is_spotify_intent(text: str) -> bool:
    lower = text.lower()
    if any(kw in lower for kw in _SPOTIFY_KEYWORDS):
        return True
    return bool(re.match(r"^(play|put on)\s+\S", lower))
_SEARCH_PHRASES = {
    "search for", "look up", "look it up", "quick answer", "find information",
    "google that", "search that",
}
_WIKIPEDIA_PHRASES = {
    "wikipedia", "wiki",
}


def _extract_news_topic(user_text: str) -> str | None:
    lower = user_text.lower()
    for marker in ("about ", "on ", "regarding ", "related to "):
        idx = lower.find("news " + marker)
        if idx != -1:
            topic = user_text[idx + len("news ") + len(marker):].strip()
            return topic[:30].strip() or None
    return None


def _extract_search_query(user_text: str) -> str:
    lower = user_text.lower()
    for marker in ("search for ", "look up ", "find information about ", "google ", "quick answer for "):
        idx = lower.find(marker)
        if idx != -1:
            return user_text[idx + len(marker):].strip()
    return user_text.strip()


def _extract_wikipedia_topic(user_text: str) -> str:
    text = user_text.strip().rstrip("?")
    # Remove "on wikipedia/wiki" suffix or inline occurrence
    text = re.sub(r"\bon\s+wiki(?:pedia)?\b", "", text, flags=re.IGNORECASE)
    # Remove "wikipedia/wiki" prefix patterns like "wikipedia for", "wiki article on"
    text = re.sub(r"\bwiki(?:pedia)?\b(?:\s+(?:article\s+(?:on|about)|for))?\s*", "", text, flags=re.IGNORECASE)
    # Strip common leading search verbs
    for prefix in ("search for", "look up", "tell me about", "find information about",
                   "what is", "what are", "who is", "who was"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip(" ,")


def _parse_spotify_command(user_text: str) -> dict:
    lower = user_text.lower()

    if any(p in lower for p in ("pause", "stop the music", "stop music")):
        return {"action": "pause"}
    if any(p in lower for p in ("next", "skip")):
        return {"action": "next"}
    if any(p in lower for p in ("previous", "go back", "last song")):
        return {"action": "previous"}
    if any(p in lower for p in ("louder", "volume up", "turn it up", "turn up")):
        return {"action": "volume_up"}
    if any(p in lower for p in ("quieter", "volume down", "turn it down", "turn down")):
        return {"action": "volume_down"}

    m = re.search(r"(?:set volume|set it) to (\d+)", lower)
    if m:
        return {"action": "set_volume", "query": m.group(1)}

    if "shuffle off" in lower or "disable shuffle" in lower:
        return {"action": "shuffle_off"}
    if "shuffle" in lower:
        return {"action": "shuffle_on"}

    if "repeat off" in lower or "stop repeating" in lower:
        return {"action": "repeat_off"}
    if any(p in lower for p in ("repeat this", "repeat track", "loop this")):
        return {"action": "repeat_track"}
    if "repeat" in lower:
        return {"action": "repeat_context"}

    if any(p in lower for p in ("like this", "like the song", "save this song", "heart this")):
        return {"action": "like_song"}
    if any(p in lower for p in ("unlike", "unsave", "remove from liked")):
        return {"action": "unlike_song"}

    if "recently played" in lower or "what have i been listening" in lower:
        return {"action": "recently_played"}

    if any(p in lower for p in ("what's playing", "current song", "now playing",
                                 "who sings", "what song", "what music")):
        return {"action": "current"}

    if any(p in lower for p in ("add to playlist", "save to playlist", "add this to my")):
        return {"action": "add_to_playlist"}

    normalised = lower.replace(",", " ").replace("  ", " ")
    for marker in ("play a song called ", "put on some ", "put on ", "play some ", "play "):
        idx = normalised.find(marker)
        if idx != -1:
            query = user_text[idx + len(marker):].strip().lstrip(", ")
            if query:
                return {"action": "play", "query": query}

    return {"action": "resume"}


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Sam] Error: ANTHROPIC_API_KEY not set. Create a .env file with your key.")
        sys.exit(1)

    currents_api_key = os.environ.get("CURRENTS_API_KEY")
    if not currents_api_key:
        print("[Sam] Warning: CURRENTS_API_KEY not set. News queries will not work.")

    spotify_id = os.environ.get("SPOTIFY_CLIENT_ID")
    spotify_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    spotify_redirect = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    sp = None
    if spotify_id and spotify_secret:
        print("[Sam] Connecting to Spotify (browser may open for first-time auth)...")
        sp = get_spotify_client(spotify_id, spotify_secret, spotify_redirect)
        print("[Sam] Spotify connected." if sp else "[Sam] Warning: Spotify connection failed.")
    else:
        print("[Sam] Warning: SPOTIFY_CLIENT_ID/SECRET not set. Spotify queries will not work.")

    if not os.path.exists("credentials.json"):
        print("[Sam] Warning: credentials.json not found. Google Calendar will not work.")

    location = get_location()
    if "error" in location:
        print(f"[Sam] Warning: could not auto-detect location — {location['error']}")
    else:
        print(f"[Sam] Location detected: {location['city']}, {location['country']}")

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

                lower_text = user_text.lower()

                # Weather intent
                if any(phrase in lower_text for phrase in _WEATHER_PHRASES):
                    if "error" in location:
                        augmented = f"[System: Location unavailable — {location['error']}] The user asked: {user_text}"
                    else:
                        weather = get_weather(location["lat"], location["lon"])
                        if "error" in weather:
                            augmented = f"[System: Weather lookup failed — {weather['error']}] The user asked: {user_text}"
                        else:
                            c = weather["current"]
                            forecast_parts = "; ".join(
                                f"{d['date']}: high={d['high_f']}°F, low={d['low_f']}°F, "
                                f"precip={d['precipitation_in']}in, UV max={d['uv_index_max']}, "
                                f"wind max={d['wind_speed_max_mph']}mph, code={d['weather_code']}"
                                for d in weather["forecast"]
                            )
                            augmented = (
                                f"[System: Current weather in {location['city']} — "
                                f"temp={c['temperature_f']}°F, feels like={c['feels_like_f']}°F, "
                                f"humidity={c['humidity_pct']}%, precipitation={c['precipitation_in']}in, "
                                f"wind={c['wind_speed_mph']}mph, UV index={c['uv_index']}, "
                                f"WMO code={c['weather_code']}, is_day={c['is_day']}. "
                                f"4-day forecast: {forecast_parts}] The user asked: {user_text}"
                            )
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # News intent
                elif any(phrase in lower_text for phrase in _NEWS_PHRASES):
                    if not currents_api_key:
                        augmented = f"[System: News unavailable — CURRENTS_API_KEY not configured.] The user asked: {user_text}"
                    else:
                        topic = _extract_news_topic(user_text)
                        articles = get_news(currents_api_key, topic=topic)
                        if articles and "error" in articles[0]:
                            augmented = f"[System: News lookup failed — {articles[0]['error']}] The user asked: {user_text}"
                        else:
                            items = "; ".join(f"'{a['title']}' ({a['source']})" for a in articles)
                            topic_note = f" about {topic}" if topic else ""
                            augmented = f"[System: Latest news{topic_note} — {items}] The user asked: {user_text}"
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # Calendar intent
                elif any(phrase in lower_text for phrase in _CALENDAR_PHRASES):
                    events = get_calendar_events()
                    if not events:
                        augmented = f"[System: No upcoming calendar events found.] The user asked: {user_text}"
                    elif "error" in events[0]:
                        augmented = f"[System: Calendar lookup failed — {events[0]['error']}] The user asked: {user_text}"
                    else:
                        items = "; ".join(
                            f"'{e['summary']}' at {e['start']}" + (f" in {e['location']}" if e['location'] else "")
                            for e in events
                        )
                        augmented = f"[System: Upcoming calendar events — {items}] The user asked: {user_text}"
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # Spotify intent
                elif _is_spotify_intent(lower_text):
                    if not sp:
                        augmented = f"[System: Spotify unavailable — not configured.] The user asked: {user_text}"
                    else:
                        cmd = _parse_spotify_command(user_text)
                        result = spotify_control(sp, cmd["action"], cmd.get("query"))
                        if "error" in result:
                            augmented = f"[System: Spotify error — {result['error']}] The user asked: {user_text}"
                        elif "success" in result:
                            augmented = f"[System: Spotify — {result['success']}] The user asked: {user_text}"
                        else:
                            state = "playing" if result["is_playing"] else "paused"
                            augmented = (
                                f"[System: Currently {state} on Spotify — '{result['track']}' "
                                f"by {result['artist']} from '{result['album']}'] The user asked: {user_text}"
                            )
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # Wikipedia (checked before general search to catch "look up X on wikipedia")
                elif any(phrase in lower_text for phrase in _WIKIPEDIA_PHRASES):
                    topic = _extract_wikipedia_topic(user_text)
                    result = get_wikipedia_summary(topic)
                    if "error" in result:
                        augmented = f"[System: Wikipedia lookup failed — {result['error']}] The user asked: {user_text}"
                    else:
                        summary = result["summary"][:800]
                        augmented = f"[System: Wikipedia — '{result['title']}': {summary}] The user asked: {user_text}"
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # DuckDuckGo web search
                elif any(phrase in lower_text for phrase in _SEARCH_PHRASES):
                    query = _extract_search_query(user_text)
                    result = get_duckduckgo_answer(query)
                    if "error" in result:
                        augmented = f"[System: No search results found for '{query}' — answer from your own knowledge.] The user asked: {user_text}"
                    else:
                        source_note = f" ({result['source']})" if result.get("source") else ""
                        augmented = f"[System: Web search results{source_note} — {result['answer']}] The user asked: {user_text}"
                    print("[Sam] ", end="", flush=True)
                    response = speaker.speak_streaming(conversation.stream_tokens(augmented))
                    print(f"{response}\n")

                # General Claude fallback
                else:
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
