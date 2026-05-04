"""Prompt loader — reads .txt templates from the prompts/ directory."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name (e.g. 'planner' reads prompts/planner.txt)."""
    path = _HERE / f"{name}.txt"
    return path.read_text(encoding="utf-8")
