"""LLM factory — ChatOpenAI instances wired to DeepSeek API / LLM 工厂——返回接入 DeepSeek API 的 ChatOpenAI 实例。

Every text-reasoning node (planner / verifier / synthesizer) calls get_llm()
with its role name; the factory selects the correct model from config.yaml.
"""

from langchain_openai import ChatOpenAI

from .config import CONFIG, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


_ROLE_TO_MODEL_KEY = {
    "planner": "planner_model",
    "verifier": "verifier_model",
    "synthesizer": "synthesizer_model",
}


def has_llm_key() -> bool:
    """Return True if the DEEPSEEK_API_KEY env var is set / 如果设置了 DEEPSEEK_API_KEY 环境变量则返回 True。"""
    return bool(DEEPSEEK_API_KEY)


def get_llm(role: str = "planner", **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI instance for the given role / 为给定角色构建ChatOpenAI实例。

    Role must be one of 'planner' / 'verifier' / 'synthesizer'. Model name and
    default temperature are read from config.yaml[llm] / 角色必须为'planner'/'verifier'/'synthesizer'之一，模型名称和默认温度从config.yaml[llm]读取。
    """
    if role not in _ROLE_TO_MODEL_KEY:
        raise ValueError(f"Unknown LLM role: {role}")
    model = CONFIG["llm"][_ROLE_TO_MODEL_KEY[role]]
    return ChatOpenAI(
        model=model,
        api_key=DEEPSEEK_API_KEY or "EMPTY",
        base_url=DEEPSEEK_BASE_URL,
        temperature=kwargs.pop("temperature", CONFIG["llm"].get("temperature", 0.0)),
        timeout=kwargs.pop("timeout", 60),
        max_retries=kwargs.pop("max_retries", 3),
        **kwargs,
    )
