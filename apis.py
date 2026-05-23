from __future__ import annotations

import datetime
import os.path
import re

import requests

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENTS_URL = "https://api.currentsapi.services/v1/latest-news"
_IP_API_URL = "http://ip-api.com/json/"
_REQUEST_TIMEOUT = 8


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
            "precipitation_sum",
            "weather_code",
            "uv_index_max",
            "wind_speed_10m_max",
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
                "precipitation_in": d["precipitation_sum"][i],
                "weather_code": d["weather_code"][i],
                "uv_index_max": d["uv_index_max"][i],
                "wind_speed_max_mph": d["wind_speed_10m_max"][i],
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


def get_news(api_key: str, topic: str | None = None, max_articles: int = 5) -> list[dict]:
    """
    Fetch latest headlines from Currents API (free tier, API key required).
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


_DUCKDUCKGO_URL = "https://api.duckduckgo.com/"


def get_duckduckgo_answer(query: str) -> dict:
    """
    Fetch an instant answer from DuckDuckGo (free, no API key).
    Returns {"answer", "source"} or {"error": str}.
    """
    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    try:
        resp = requests.get(_DUCKDUCKGO_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("Answer") or data.get("AbstractText") or data.get("Definition") or ""
        if not answer:
            return {"error": "No instant answer found"}
        return {"answer": answer, "source": data.get("AnswerType") or data.get("AbstractSource") or ""}
    except requests.RequestException as exc:
        return {"error": f"Search unavailable: {exc}"}
    except (KeyError, ValueError) as exc:
        return {"error": f"Unexpected search data: {exc}"}


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
                "playlist-modify-public "
                "playlist-modify-private "
                "playlist-read-private"
            ),
            open_browser=True,
        ))
        sp.current_user()
        return sp
    except Exception:
        return None


def _get_device_id(sp) -> str | None:
    """
    Return the ID of the active Spotify device.
    If no device is active, transfer playback to the first available one.
    Returns None if no devices are found at all.
    """
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    for d in devices:
        if d["is_active"]:
            return d["id"]
    first_id = devices[0]["id"]
    sp.transfer_playback(first_id, force_play=False)
    return first_id


def spotify_control(sp, action: str, query: str | None = None) -> dict:
    """
    Execute a Spotify playback action.
    action: "pause", "resume", "next", "previous", "volume_up", "volume_down",
            "shuffle", "play" (with query), "current", "add_to_playlist"
    Returns {"success": str}, a track info dict, or {"error": str}.
    """
    try:
        device_id = _get_device_id(sp)
        if device_id is None:
            return {"error": "No Spotify device found — open the Spotify app on any device first"}
        if action == "pause":
            sp.pause_playback(device_id=device_id)
            return {"success": "Playback paused"}
        if action == "resume":
            sp.start_playback(device_id=device_id)
            return {"success": "Playback resumed"}
        if action == "next":
            sp.next_track(device_id=device_id)
            return {"success": "Skipped to next track"}
        if action == "previous":
            sp.previous_track(device_id=device_id)
            return {"success": "Went back to previous track"}
        if action == "volume_up":
            pb = sp.current_playback()
            current_vol = (pb or {}).get("device", {}).get("volume_percent") or 50
            vol = min(100, current_vol + 20)
            sp.volume(vol, device_id=device_id)
            return {"success": f"Volume set to {vol}%"}
        if action == "volume_down":
            pb = sp.current_playback()
            current_vol = (pb or {}).get("device", {}).get("volume_percent") or 50
            vol = max(0, current_vol - 20)
            sp.volume(vol, device_id=device_id)
            return {"success": f"Volume set to {vol}%"}
        if action == "shuffle":
            sp.shuffle(True, device_id=device_id)
            return {"success": "Shuffle enabled"}
        if action == "play" and query:
            q_lower = query.lower()

            # "Song by Artist" → use Spotify field filters for precision
            if " by " in q_lower:
                parts = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.IGNORECASE)
                formatted = f"track:{parts[0].strip()} artist:{parts[1].strip()}"
                res = sp.search(q=formatted, type="track", limit=1)
                tracks = res.get("tracks", {}).get("items", [])
                if not tracks:  # fallback to plain search
                    res = sp.search(q=query, type="track", limit=1)
                    tracks = res.get("tracks", {}).get("items", [])
                if tracks:
                    sp.start_playback(device_id=device_id, uris=[tracks[0]["uri"]])
                    return {"success": f"Playing '{tracks[0]['name']}' by {tracks[0]['artists'][0]['name']}"}
                return {"error": f"Nothing found for '{query}'"}

            # "my ... playlist" or query ends with "playlist" → search user playlists
            if "playlist" in q_lower or q_lower.startswith("my "):
                pl_name = q_lower.removeprefix("my ").removesuffix(" playlist").strip()
                playlists = sp.current_user_playlists(limit=20)
                for pl in playlists.get("items", []):
                    if pl_name in pl["name"].lower():
                        sp.start_playback(device_id=device_id, context_uri=pl["uri"])
                        return {"success": f"Playing playlist '{pl['name']}'"}
                return {"error": f"No playlist found matching '{pl_name}'"}

            # Try exact artist match first → play artist context
            artist_res = sp.search(q=query, type="artist", limit=1)
            artists = artist_res.get("artists", {}).get("items", [])
            if artists and artists[0]["name"].lower() == q_lower:
                sp.start_playback(device_id=device_id, context_uri=artists[0]["uri"])
                return {"success": f"Playing {artists[0]['name']}"}

            # Fall back to track search
            track_res = sp.search(q=query, type="track", limit=1)
            tracks = track_res.get("tracks", {}).get("items", [])
            if tracks:
                sp.start_playback(device_id=device_id, uris=[tracks[0]["uri"]])
                return {"success": f"Playing '{tracks[0]['name']}' by {tracks[0]['artists'][0]['name']}"}
            return {"error": f"Nothing found for '{query}'"}
        if action == "play":
            sp.start_playback(device_id=device_id)
            return {"success": "Playback resumed"}
        if action == "current":
            track = sp.current_user_playing_track()
            if track and track.get("item"):
                item = track["item"]
                return {
                    "track": item["name"],
                    "artist": ", ".join(a["name"] for a in item["artists"]),
                    "album": item["album"]["name"],
                    "is_playing": track["is_playing"],
                }
            return {"error": "Nothing is currently playing"}
        if action == "add_to_playlist":
            if query:
                results = sp.search(q=query, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                if not tracks:
                    return {"error": f"No track found for '{query}'"}
                track_uri = tracks[0]["uri"]
                track_name = f"{tracks[0]['name']} by {tracks[0]['artists'][0]['name']}"
            else:
                current = sp.current_user_playing_track()
                if not current or not current.get("item"):
                    return {"error": "Nothing is currently playing to add"}
                track_uri = current["item"]["uri"]
                track_name = f"{current['item']['name']} by {current['item']['artists'][0]['name']}"
            playlists = sp.current_user_playlists(limit=10)
            items = playlists.get("items", [])
            if not items:
                return {"error": "No playlists found on your account"}
            playlist = items[0]
            sp.playlist_add_items(playlist["id"], [track_uri])
            return {"success": f"Added '{track_name}' to playlist '{playlist['name']}'"}
        return {"error": f"Unknown Spotify action: {action}"}
    except Exception as exc:
        return {"error": f"Spotify error: {exc}"}
