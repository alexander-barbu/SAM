from __future__ import annotations

from collections.abc import Callable, Iterator

import anthropic

from config import CLAUDE_MODEL, MAX_HISTORY_TURNS, SYSTEM_PROMPT


class ConversationManager:
    """
    Maintains a rolling conversation history and streams responses from Claude.
    System prompt is passed separately from message history on every call.
    """

    def __init__(self, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict[str, str]] = []

    def ask(self, user_text: str) -> str:
        """Convenience wrapper — collects the full streamed response and returns it."""
        return "".join(self.stream_tokens(user_text))

    def stream_tokens(self, user_text: str) -> Iterator[str]:
        """
        Yield text tokens from Claude as they arrive.
        History is updated with the complete response once streaming ends.
        """
        self._append("user", user_text)
        self._trim()

        full_response = ""
        with self.client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=self.history,
        ) as stream:
            for delta in stream.text_stream:
                full_response += delta
                yield delta

        self._append("assistant", full_response)

    def reset(self) -> None:
        """Clear all conversation history."""
        self.history.clear()

    def _append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _trim(self) -> None:
        """
        Keep only the most recent MAX_HISTORY_TURNS complete turns.
        Always trim in pairs so the message list stays properly alternating.
        """
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]
