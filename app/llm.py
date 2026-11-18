"""The "LLM" of this project is our own GPT, trained locally: no API, no key, no network.

    from app.llm import MedicalGPT
    gpt = MedicalGPT()                        # loads models/medgpt.pt (or $MODEL_PATH)
    gpt.complete("The femur is", length=40)
    gpt.define("pericardium")
"""
import os
from pathlib import Path

from app import core


def load_env() -> None:
    """Read optional KEY=VALUE lines from .env without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def model_path() -> Path:
    load_env()
    return Path(os.environ.get("MODEL_PATH") or core.MODEL_DIR / "medgpt.pt")


class MedicalGPT:
    """Load the trained checkpoint once; complete phrases and define terms."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else model_path()
        self.model, self.tok, self.info = core.load_checkpoint(self.path)   # ModelNotFoundError if missing

    def _prep(self, text: str) -> str:
        text = core.normalize_chars(text)            # curly quotes, accents, ½ ... -> the corpus alphabet
        bad = self.tok.unknown_chars(text)
        if bad:
            raise core.UnknownCharacterError(bad)
        return text

    def complete(self, prompt: str, length: int = 60, temperature: float = 0.8, top_k: int | None = 40,
                 top_p: float | None = 0.95, seed: int | None = None, stop_at_sentence: bool = False) -> str:
        return core.complete(self.model, self.tok, self._prep(prompt), length, temperature, top_k, top_p,
                             seed, stop_at_sentence)

    def define(self, term: str, length: int = 40, temperature: float = 0.7, top_k: int | None = 40,
               top_p: float | None = 0.9, seed: int | None = None) -> str:
        return core.define(self.model, self.tok, self._prep(term), length, temperature=temperature,
                           top_k=top_k, top_p=top_p, seed=seed)
