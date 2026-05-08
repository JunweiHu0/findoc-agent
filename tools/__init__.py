# Tool auto-registration — importing registry triggers register() for all built-in tools.
from tools.registry import REGISTRY, dispatch, get_tools_for_prompt, register, ToolSpec  # noqa: F401
