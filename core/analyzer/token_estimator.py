from __future__ import annotations


class TokenEstimator:
    """Estimate token count from input text.

    Uses a fast character-count heuristic by default.
    Falls back to tiktoken if available for more accurate estimation.
    """

    # Approximate: 4 chars per token (English prose)
    _CHARS_PER_TOKEN = 4.0

    def __init__(self) -> None:
        self._tiktoken_enc = None
        self._tiktoken_available = False
        self._try_load_tiktoken()

    def _try_load_tiktoken(self) -> None:
        try:
            import tiktoken
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_available = True
        except ImportError:
            pass

    def estimate(self, text: str) -> int:
        if self._tiktoken_available and self._tiktoken_enc:
            try:
                return len(self._tiktoken_enc.encode(text))
            except Exception:
                pass
        # Fallback: character heuristic
        return max(1, int(len(text) / self._CHARS_PER_TOKEN))
