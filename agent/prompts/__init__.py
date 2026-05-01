from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    path = _HERE / f"{name}.txt"
    return path.read_text(encoding="utf-8")
