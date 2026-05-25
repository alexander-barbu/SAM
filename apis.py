from __future__ import annotations

import datetime
import os.path
import re
import time

import requests

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENTS_URL = "https://api.currentsapi.services/v1/latest-news"
_IP_API_URL = "http://ip-api.com/json/"
_REQUEST_TIMEOUT = 8


# ── Location ──────────────────────────────────────────────────────────────────

def get_location() -> dict:
    """
    Auto-detect lat/lon/city/country from the current public IP via ip-api.com.
    Returns {"lat", "lon", "city", "country"} or {"error": str}.
    """
    try:
        resp = requests.get(_IP_API_URL, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            return {"error": f"ip-api returned status: {data.get('message', 'unknown')}"}
        return {
            "lat": data["lat"],
            "lon": data["lon"],
            "city": data.get("city", "Unknown"),
            "country": data.get("country", "Unknown"),
        }
    except requests.RequestException as exc:
        return {"error": f"Location service unavailable: {exc}"}
    except (KeyError, ValueError) as exc:
        return {"error": f"Unexpected location data format: {exc}"}


# ── Weather ───────────────────────────────────────────────────────────────────

def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather + 3-day daily forecast from Open-Meteo (free, no API key).
    Units: Fahrenheit, mph, inches.
    Returns {"current": {...}, "forecast": [...]} or {"error": str}.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "is_day",
            "uv_index",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "weather_code",
            "uv_index_max",
            "wind_speed_10m_max",
            "sunrise",
            "sunset",
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "forecast_days": 4,
        "timezone": "auto",
    }
    try:
        resp = requests.get(_OPEN_METEO_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        c = data["current"]
        d = data["daily"]
        forecast = [
            {
                "date": d["time"][i],
                "high_f": d["temperature_2m_max"][i],
                "low_f": d["temperature_2m_min"][i],
                "feels_like_high_f": d["apparent_temperature_max"][i],
                "feels_like_low_f": d["apparent_temperature_min"][i],
                "precipitation_in": d["precipitation_sum"][i],
                "precipitation_probability_pct": d["precipitation_probability_max"][i],
                "weather_code": d["weather_code"][i],
                "uv_index_max": d["uv_index_max"][i],
                "wind_speed_max_mph": d["wind_speed_10m_max"][i],
                "sunrise": d["sunrise"][i],
                "sunset": d["sunset"][i],
            }
            for i in range(len(d["time"]))
        ]
        return {
            "current": {
                "temperature_f": c["temperature_2m"],
                "feels_like_f": c["apparent_temperature"],
                "humidity_pct": c["relative_humidity_2m"],
                "precipitation_in": c["precipitation"],
                "wind_speed_mph": c["wind_speed_10m"],
                "wind_direction_deg": c["wind_direction_10m"],
                "weather_code": c["weather_code"],
                "uv_index": c["uv_index"],
                "is_day": c["is_day"],
            },
            "forecast": forecast,
        }
    except requests.RequestException as exc:
        return {"error": f"Weather service unavailable: {exc}"}
    except (KeyError, ValueError) as exc:
        return {"error": f"Unexpected weather data format: {exc}"}


# ── News ──────────────────────────────────────────────────────────────────────

def get_news(api_key: str, topic: str | None = None, max_articles: int = 5) -> list[dict]:
    """
    Fetch latest news from Currents API (free tier, API key required).
    Returns list of {"title", "description", "source", "published"} or [{"error": str}].
    """
    params: dict = {"apiKey": api_key, "language": "en"}
    if topic:
        params["keywords"] = topic
    try:
        resp = requests.get(_CURRENTS_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        articles = resp.json().get("news", [])[:max_articles]
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", ""),
                "published": a.get("published", ""),
            }
            for a in articles
        ]
    except requests.RequestException as exc:
        return [{"error": f"News service unavailable: {exc}"}]
    except (KeyError, ValueError) as exc:
        return [{"error": f"Unexpected news data format: {exc}"}]


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def get_duckduckgo_answer(query: str) -> dict:
    """
    Search the web via DuckDuckGo and return a summary of the top results.
    Returns {"answer", "source"} or {"error": str}.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return {"error": "No results found"}
        snippets = " | ".join(r["body"] for r in results if r.get("body"))
        source = results[0].get("href", "")
        return {"answer": snippets, "source": source}
    except Exception as exc:
        return {"error": f"Search unavailable: {exc}"}


# ── Wikipedia ─────────────────────────────────────────────────────────────────

_WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
_WIKIPEDIA_HEADERS = {"User-Agent": "SAM-VoiceAssistant/1.0"}


def get_wikipedia_summary(query: str) -> dict:
    """
    Search Wikipedia and return the summary for the top result (free, no API key).
    Returns {"title", "summary"} or {"error": str}.
    """
    try:
        search_resp = requests.get(
            _WIKIPEDIA_SEARCH_URL,
            params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
            headers=_WIKIPEDIA_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        search_resp.raise_for_status()
        results = search_resp.json()
        if not results[1]:
            return {"error": f"No Wikipedia article found for '{query}'"}
        title = results[1][0]
        summary_resp = requests.get(
            f"{_WIKIPEDIA_SUMMARY_URL}/{requests.utils.quote(title)}",
            headers=_WIKIPEDIA_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()
        return {"title": data.get("title", title), "summary": data.get("extract", "")}
    except requests.RequestException as exc:
        return {"error": f"Wikipedia unavailable: {exc}"}
    except (KeyError, ValueError, IndexError) as exc:
        return {"error": f"Unexpected Wikipedia data: {exc}"}


# ── Google Calendar ───────────────────────────────────────────────────────────

_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def get_calendar_events(max_results: int = 5) -> list[dict]:
    """
    Fetch upcoming Google Calendar events (OAuth2, requires credentials.json).
    Returns list of {"summary", "start", "end", "location"} or [{"error": str}].
    Token is stored in token.json and refreshed automatically.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", _CALENDAR_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", _CALENDAR_SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as f:
                f.write(creds.to_json())

        service = build("calendar", "v3", credentials=creds)
        now = datetime.datetime.utcnow().isoformat() + "Z"
        result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        if not events:
            return []
        return [
            {
                "summary": e.get("summary", "No title"),
                "start": e["start"].get("dateTime", e["start"].get("date", "")),
                "end": e["end"].get("dateTime", e["end"].get("date", "")),
                "location": e.get("location", ""),
            }
            for e in events
        ]
    except FileNotFoundError:
        return [{"error": "credentials.json not found — download it from Google Cloud Console"}]
    except Exception as exc:
        return [{"error": f"Google Calendar error: {exc}"}]


# ── Spotify ───────────────────────────────────────────────────────────────────

def get_spotify_client(client_id: str, client_secret: str, redirect_uri: str):
    """
    Create and return an authenticated Spotipy client, opening a browser on first run.
    Returns the client, or None on failure.
    """
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=(
                "user-read-playback-state "
                "user-modify-playback-state "
                "user-read-currently-playing "
                "user-library-modify "        # FIX: needed for liking songs
                "user-library-read "          # FIX: needed for checking liked status
                "playlist-modify-public "
                "playlist-modify-private "
                "playlist-read-private "
                "user-read-recently-played"   # FIX: needed for recently played
            ),
            open_browser=True,
        ))
        sp.current_user()
        return sp
    except Exception as exc:
        print(f"[Sam] Spotify auth failed: {exc}")
        return None


def _get_active_device(sp) -> tuple[str | None, bool]:
    """
    Return (device_id, was_transferred).
    If no device is active, transfer playback to the first available one
    and wait briefly for the transfer to complete.
    Returns (None, False) if no devices found at all.
    """
    try:
        devices = sp.devices().get("devices", [])
    except Exception:
        return None, False

    if not devices:
        return None, False

    # Return active device immediately if found
    for d in devices:
        if d["is_active"]:
            return d["id"], False

    # No active device — transfer to first available and wait
    first_id = devices[0]["id"]
    try:
        sp.transfer_playback(first_id, force_play=False)
        time.sleep(1.0)
    except Exception as exc:
        print(f"[Sam] Spotify device transfer failed: {exc}")
    return first_id, True


def spotify_control(sp, action: str, query: str | None = None) -> dict:
    """
    Execute a Spotify playback action.

    action values:
        "pause"           — pause playback
        "resume"          — resume playback
        "next"            — skip to next track
        "previous"        — go back to previous track
        "volume_up"       — increase volume by 20%
        "volume_down"     — decrease volume by 20%
        "set_volume"      — set exact volume (query = "50" for 50%)
        "shuffle_on"      — enable shuffle
        "shuffle_off"     — disable shuffle
        "repeat_track"    — repeat current track
        "repeat_context"  — repeat playlist/album
        "repeat_off"      — turn off repeat
        "play"            — play a song/artist/playlist (query = search term)
        "current"         — what is currently playing
        "like_song"       — like/save current song to Liked Songs
        "unlike_song"     — remove current song from Liked Songs
        "recently_played" — list recently played tracks
        "add_to_playlist" — add current or searched song to a named playlist
                            (query = "playlist name" or "song name >> playlist name")

    Returns {"success": str}, a track info dict, or {"error": str}.
    """
    try:
        device_id, was_transferred = _get_active_device(sp)

        # Actions that don't need a device
        if action == "current":
            track = sp.current_user_playing_track()
            if track and track.get("item"):
                item = track["item"]
                return {
                    "track": item["name"],
                    "artist": ", ".join(a["name"] for a in item.get("artists", [])) or "Unknown",
                    "album": item.get("album", {}).get("name", "Unknown"),
                    "is_playing": track["is_playing"],
                }
            return {"error": "Nothing is currently playing"}

        if action == "recently_played":
            results = sp.current_user_recently_played(limit=5)
            items = results.get("items", [])
            if not items:
                return {"error": "No recently played tracks found"}
            tracks = [
                f"{i['track']['name']} by {i['track']['artists'][0]['name']}"
                for i in items
            ]
            return {"recently_played": tracks}

        if action == "like_song":
            track = sp.current_user_playing_track()
            if not track or not track.get("item"):
                return {"error": "Nothing is currently playing to like"}
            track_id = track["item"]["id"]
            track_name = track["item"]["name"]
            artist_name = (track["item"].get("artists") or [{}])[0].get("name", "Unknown")
            sp.current_user_saved_tracks_add([track_id])
            return {"success": f"Added '{track_name}' by {artist_name} to your Liked Songs"}

        if action == "unlike_song":
            track = sp.current_user_playing_track()
            if not track or not track.get("item"):
                return {"error": "Nothing is currently playing to unlike"}
            track_id = track["item"]["id"]
            track_name = track["item"]["name"]
            sp.current_user_saved_tracks_delete([track_id])
            return {"success": f"Removed '{track_name}' from your Liked Songs"}

        # Actions that need a device
        if device_id is None:
            return {"error": "No Spotify device found — open the Spotify app on any device first"}

        if action in ("pause", "resume"):
            pb = sp.current_playback()
            is_playing = (pb or {}).get("is_playing", False)

            if action == "pause":
                if not is_playing:
                    return {"success": "Already paused"}
                sp.pause_playback(device_id=device_id)
                return {"success": "Playback paused"}

            if action == "resume":
                if is_playing:
                    return {"success": "Already playing"}
                sp.start_playback(device_id=device_id)
                return {"success": "Playback resumed"}

        if action == "next":
            sp.next_track(device_id=device_id)
            time.sleep(0.5)
            return {"success": "Skipped to next track"}

        if action == "previous":
            sp.previous_track(device_id=device_id)
            time.sleep(0.5)
            return {"success": "Went back to previous track"}

        if action == "volume_up":
            pb = sp.current_playback()
            current_vol = (pb or {}).get("device", {}).get("volume_percent")
            if current_vol is None:
                current_vol = 50  # safe default if unreadable
            vol = min(100, current_vol + 20)
            sp.volume(int(vol), device_id=device_id)  # FIX: ensure int, not float
            return {"success": f"Volume set to {vol}%"}

        if action == "volume_down":
            pb = sp.current_playback()
            current_vol = (pb or {}).get("device", {}).get("volume_percent")
            if current_vol is None:
                current_vol = 50
            vol = max(0, current_vol - 20)
            sp.volume(int(vol), device_id=device_id)  # FIX: ensure int, not float
            return {"success": f"Volume set to {vol}%"}

        if action == "set_volume":
            try:
                vol = max(0, min(100, int(query or 50)))
            except (ValueError, TypeError):
                return {"error": "Please specify a volume between 0 and 100"}
            sp.volume(vol, device_id=device_id)
            return {"success": f"Volume set to {vol}%"}

        if action == "shuffle_on":
            sp.shuffle(True, device_id=device_id)
            return {"success": "Shuffle enabled"}

        if action == "shuffle_off":
            sp.shuffle(False, device_id=device_id)
            return {"success": "Shuffle disabled"}

        if action == "repeat_track":
            sp.repeat("track", device_id=device_id)
            return {"success": "Repeating current track"}

        if action == "repeat_context":
            sp.repeat("context", device_id=device_id)
            return {"success": "Repeating playlist or album"}

        if action == "repeat_off":
            sp.repeat("off", device_id=device_id)
            return {"success": "Repeat disabled"}

        if action == "play_playlist":
            playlists = sp.current_user_playlists(limit=50)
            items = [p for p in playlists.get("items", []) if p]
            if not items:
                return {"error": "No playlists found on your account"}
            if query:
                q_lower = query.lower()
                for pl in items:
                    if q_lower in pl["name"].lower():
                        sp.start_playback(device_id=device_id, context_uri=pl["uri"])
                        return {"success": f"Playing playlist '{pl['name']}'"}
                return {"error": f"No playlist found matching '{query}'"}
            pl = items[0]
            sp.start_playback(device_id=device_id, context_uri=pl["uri"])
            return {"success": f"Playing playlist '{pl['name']}'"}

        if action == "play" and query:
            q_lower = query.lower()

            # "Song by Artist" → use Spotify field filters for precision
            if " by " in q_lower:
                parts = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.IGNORECASE)
                formatted = f"track:{parts[0].strip()} artist:{parts[1].strip()}"
                res = sp.search(q=formatted, type="track", limit=1)
                tracks = res.get("tracks", {}).get("items", [])
                if not tracks:
                    # Fallback: broader search but only keep tracks whose artist name
                    # overlaps with what was asked — prevents totally wrong results
                    res = sp.search(q=query, type="track", limit=5)
                    queried_artist = parts[1].strip().lower()
                    tracks = [
                        t for t in res.get("tracks", {}).get("items", [])
                        if any(
                            queried_artist in a["name"].lower() or a["name"].lower() in queried_artist
                            for a in t.get("artists", [])
                        )
                    ]
                if tracks:
                    t = tracks[0]
                    artist = (t.get("artists") or [{}])[0].get("name", "Unknown")
                    sp.start_playback(device_id=device_id, uris=[t["uri"]])
                    return {"success": f"Playing '{t['name']}' by {artist}"}
                return {"error": f"Couldn't find '{parts[0].strip()}' by {parts[1].strip()} on Spotify"}

            # Liked Songs
            if q_lower in ("liked songs", "liked", "my liked songs", "saved songs", "saved"):
                try:
                    sp.start_playback(device_id=device_id, context_uri="spotify:collection:tracks")
                    return {"success": "Playing your Liked Songs"}
                except Exception:
                    saved = sp.current_user_saved_tracks(limit=50)
                    uris = [i["track"]["uri"] for i in saved.get("items", []) if i.get("track")]
                    if not uris:
                        return {"error": "No liked songs found in your library"}
                    sp.start_playback(device_id=device_id, uris=uris)
                    return {"success": "Playing your Liked Songs"}

            # Playlist search
            if "playlist" in q_lower or q_lower.startswith("my "):
                pl_name = q_lower.removeprefix("my ").removesuffix(" playlist").strip()
                playlists = sp.current_user_playlists(limit=50)  # FIX: wider search
                for pl in playlists.get("items", []):
                    if pl and pl_name in pl["name"].lower():
                        sp.start_playback(device_id=device_id, context_uri=pl["uri"])
                        return {"success": f"Playing playlist '{pl['name']}'"}
                return {"error": f"No playlist found matching '{pl_name}'"}

            # Try exact artist match first
            artist_res = sp.search(q=query, type="artist", limit=1)
            artists = artist_res.get("artists", {}).get("items", [])
            if artists and artists[0]["name"].lower() == q_lower:
                sp.start_playback(device_id=device_id, context_uri=artists[0]["uri"])
                return {"success": f"Playing {artists[0]['name']}"}

            # Fall back to track search
            track_res = sp.search(q=query, type="track", limit=1)
            tracks = track_res.get("tracks", {}).get("items", [])
            if tracks:
                t = tracks[0]
                artist = (t.get("artists") or [{}])[0].get("name", "Unknown")
                sp.start_playback(device_id=device_id, uris=[t["uri"]])
                return {"success": f"Playing '{t['name']}' by {artist}"}
            return {"error": f"Nothing found for '{query}'"}

        if action == "play":
            sp.start_playback(device_id=device_id)
            return {"success": "Playback resumed"}

        if action == "add_to_playlist":
            # FIX: support "song >> playlist name" format for targeting a specific playlist
            target_playlist_name = None
            search_query = None

            if query and ">>" in query:
                parts = query.split(">>", 1)
                search_query = parts[0].strip() or None
                target_playlist_name = parts[1].strip()
            else:
                search_query = query

            # Get the track to add
            if search_query:
                results = sp.search(q=search_query, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                if not tracks:
                    return {"error": f"No track found for '{search_query}'"}
                track_uri = tracks[0]["uri"]
                artist = (tracks[0].get("artists") or [{}])[0].get("name", "Unknown")
                track_name = f"{tracks[0]['name']} by {artist}"
            else:
                current = sp.current_user_playing_track()
                if not current or not current.get("item"):
                    return {"error": "Nothing is currently playing to add"}
                track_uri = current["item"]["uri"]
                artist = (current["item"].get("artists") or [{}])[0].get("name", "Unknown")
                track_name = f"{current['item']['name']} by {artist}"

            # FIX: find the target playlist by name, or default to first playlist
            playlists = sp.current_user_playlists(limit=50)
            items = [p for p in playlists.get("items", []) if p]  # filter None items
            if not items:
                return {"error": "No playlists found on your account"}

            playlist = None
            if target_playlist_name:
                for pl in items:
                    if target_playlist_name.lower() in pl["name"].lower():
                        playlist = pl
                        break
                if not playlist:
                    return {"error": f"No playlist found matching '{target_playlist_name}'"}
            else:
                playlist = items[0]

            sp.playlist_add_items(playlist["id"], [track_uri])
            return {"success": f"Added '{track_name}' to playlist '{playlist['name']}'"}

        return {"error": f"Unknown Spotify action: '{action}'"}

    except Exception as exc:
        # Extract Spotify's specific reason (e.g. PREMIUM_REQUIRED, NO_ACTIVE_DEVICE)
        reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
        return {"error": f"Spotify: {reason or exc}"}