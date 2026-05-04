import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG: dict = yaml.safe_load(f)


def _abs(rel: str) -> Path:
    """Resolve a relative config path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

PDF_DIR = _abs(CONFIG["ingestion"]["pdf_dir"])
PAGES_DIR = _abs(CONFIG["ingestion"]["pages_dir"])
INDEX_DIR = _abs(CONFIG["retriever"]["index_dir"])
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = ROOT / "checkpoints"

MAX_REFLEXION_ITER: int = CONFIG["agent"]["max_reflexion_iter"]
TOP_K: int = CONFIG["retriever"]["top_k"]
