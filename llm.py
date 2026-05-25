from __future__ import annotations

from collections.abc import Callable, Iterator

import anthropic

from config import CLAUDE_MODEL, MAX_HISTORY_TURNS, SYSTEM_PROMPT
from memory import MemoryManager


class ConversationManager:
    """
    Maintains a rolling conversation history and streams responses from Claude.
    System prompt is passed separately from message history on every call.
    """

    def __init__(self, api_key: str | None = None, memory: MemoryManager | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.history: list[dict[str, str]] = []
        self._memory = memory or MemoryManager()

    def ask(self, user_text: str) -> str:
        """Convenience wrapper — collects the full streamed response and returns it."""
        return "".join(self.stream_tokens(user_text))

    def stream_tokens(self, user_text: str) -> Iterator[str]:
        """
        Yield text tokens from Claude as they arrive.
        [REMEMBER: ...] tags are filtered from the stream and saved to persistent memory.
        History is updated with the complete clean response once streaming ends.
        """
        self._append("user", user_text)
        self._trim()

        raw_parts: list[str] = []
        clean_response = ""

        def _tee(source: Iterator[str]) -> Iterator[str]:
            for token in source:
                raw_parts.append(token)
                yield token

        with self.client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=self._build_system(),
            messages=self.history,
        ) as stream:
            for clean_token in self._filtered_stream(_tee(stream.text_stream)):
                clean_response += clean_token
                yield clean_token

        self._memory.extract_and_save("".join(raw_parts))
        self._append("assistant", clean_response)

    def reset(self) -> None:
        """Clear conversation history (memory is preserved)."""
        self.history.clear()

    def _build_system(self) -> list[dict]:
        blocks: list[dict] = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]
        mem = self._memory.as_context_block()
        if mem:
            blocks.append({"type": "text", "text": mem})
        return blocks

    def _filtered_stream(self, source: Iterator[str]) -> Iterator[str]:
        """Yield tokens with [REMEMBER: ...] and [System: ...] spans removed."""
        OPENERS = ["[REMEMBER:", "[SYSTEM:"]
        state = "NORMAL"
        buf = ""

        def _matches_any_opener(s: str) -> bool:
            return any(s.upper().startswith(o) for o in OPENERS)

        def _could_start_opener(s: str) -> bool:
            return any(o.startswith(s.upper()) for o in OPENERS)

        for token in source:
            if state == "NORMAL":
                if "[" in token:
                    idx = token.index("[")
                    yield token[:idx]
                    buf = token[idx:]
                    state = "TENTATIVE"
                else:
                    yield token

            elif state == "TENTATIVE":
                buf += token
                if _matches_any_opener(buf):
                    state = "SUPPRESSING"
                    buf = ""
                elif not _could_start_opener(buf):
                    yield buf
                    buf = ""
                    state = "NORMAL"

            elif state == "SUPPRESSING":
                if "]" in token:
                    after = token[token.index("]") + 1:]
                    if after:
                        yield after
                    state = "NORMAL"

        if buf:
            yield buf

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
