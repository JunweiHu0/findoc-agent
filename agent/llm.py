from langchain_openai import ChatOpenAI

from .config import CONFIG, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


_ROLE_TO_MODEL_KEY = {
    "planner": "planner_model",
    "verifier": "verifier_model",
    "synthesizer": "synthesizer_model",
}


def has_llm_key() -> bool:
    return bool(DEEPSEEK_API_KEY)


def get_llm(role: str = "planner", **kwargs) -> ChatOpenAI:
    if role not in _ROLE_TO_MODEL_KEY:
        raise ValueError(f"Unknown LLM role: {role}")
    model = CONFIG["llm"][_ROLE_TO_MODEL_KEY[role]]
    return ChatOpenAI(
        model=model,
        api_key=DEEPSEEK_API_KEY or "EMPTY",
        base_url=DEEPSEEK_BASE_URL,
        temperature=kwargs.pop("temperature", CONFIG["llm"].get("temperature", 0.0)),
        **kwargs,
    )
