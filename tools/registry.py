"""Tool Registry (P21) — typed, self-describing tool dispatch.

Each tool registers a ToolSpec: name, human description (for the planner),
typed params_schema / output_schema (both pydantic), and a callable.

The planner prompt receives the registry summary so it can emit explicit
tool_calls; the executor dispatches via `dispatch()` and validates outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel
from loguru import logger


@dataclass
class ToolSpec:
    name: str
    description: str                    # human-readable, injected into planner prompt
    params_schema: type[BaseModel]      # pydantic input model
    output_schema: type[BaseModel]      # pydantic output model
    callable: Callable                  # the actual function
    category: str = "general"           # "retrieval" | "reading" | "compute" | "resolution"

    def params_schema_json(self) -> dict:
        """Return a simplified JSON-schema-ish dict for the planner prompt."""
        return _pydantic_to_prompt_schema(self.params_schema)


REGISTRY: dict[str, ToolSpec] = {}

# Simple input/output models for tools that don't have complex args
class _RetrieveInput(BaseModel):
    query: str
    top_k: int = 5
    doc_filter: Optional[list[str]] = None

class _RetrieveOutput(BaseModel):
    pages: list[dict]  # [{doc_id, page_num, score, image_path}]

class _ReadPageInput(BaseModel):
    image_path: str
    instruction: str

class _ReadPageOutput(BaseModel):
    extracted_text: str

class _CalculateInput(BaseModel):
    expression: str

class _CalculateOutput(BaseModel):
    value: float

class _DisambiguateInput(BaseModel):
    conflict_topic: str
    fact_texts: list[str]

class _DisambiguateOutput(BaseModel):
    resolved: bool
    authoritative_fact: dict
    explanation: str


def _pydantic_to_prompt_schema(model: type[BaseModel]) -> dict:
    """Convert a pydantic model's fields to a compact dict for the prompt."""
    out: dict[str, str] = {}
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        type_str = _type_name(annotation)
        if field_info.default is not None and field_info.default != field_info.get_default():
            pass  # has default
        out[name] = type_str
    return out


def _type_name(annotation) -> str:
    """Short type name for prompt display."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        args = getattr(annotation, "__args__", ())
        inner = _type_name(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin is dict:
        return "dict"
    name = getattr(annotation, "__name__", str(annotation))
    return name


def register(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec
    logger.debug(f"registered tool: {spec.name} ({spec.category})")


def get_tools_for_prompt() -> str:
    """Format the registry as a planner-prompt section."""
    if not REGISTRY:
        return "(no tools registered)"
    lines: list[str] = []
    for name, spec in sorted(REGISTRY.items()):
        params = spec.params_schema_json()
        params_str = ", ".join(f"{k}: {v}" for k, v in params.items()) or "none"
        lines.append(f"- **{name}** ({spec.category}): {spec.description}")
        lines.append(f"  params: {{{params_str}}}")
    return "\n".join(lines)


def dispatch(tool_name: str, args: dict) -> Any:
    """Invoke a tool by name; validate output against output_schema."""
    spec = REGISTRY.get(tool_name)
    if spec is None:
        raise KeyError(f"Unknown tool: {tool_name}. Available: {list(REGISTRY.keys())}")

    # Validate input
    parsed_input = spec.params_schema(**args)

    # Call the tool — pass args as kwargs
    try:
        if isinstance(parsed_input, BaseModel):
            raw = spec.callable(**parsed_input.model_dump())
        else:
            raw = spec.callable(**args)
    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}")
        raise

    # Validate output
    try:
        validated = spec.output_schema(**raw) if isinstance(raw, dict) else raw
        return validated
    except Exception as e:
        logger.warning(f"Tool '{tool_name}' output validation failed ({e}); returning raw output")
        return raw


# ---------------------------------------------------------------------------
# Auto-register built-in tools on import
# ---------------------------------------------------------------------------

def _register_builtins():
    """Register all built-in tools. Called at module import time."""
    # Avoid circular imports — import tools lazily inside each callable wrapper
    from tools.calculator import calculate as _calc
    from tools.colpali_tool import colpali_retrieve as _retrieve
    from tools.vlm_tool import vlm_read_page as _vlm_read
    from tools.disambiguate import disambiguate_caliber as _disambig

    register(ToolSpec(
        name="retrieve_pages",
        description="Multi-vector ColQwen2 retrieval. Searches indexed financial documents for pages relevant to the query. Returns top-k page hits with doc_id, page_num, and relevance score.",
        params_schema=_RetrieveInput,
        output_schema=_RetrieveOutput,
        callable=lambda query, top_k=5, doc_filter=None: {
            "pages": [
                {"doc_id": h.doc_id, "page_num": h.page_num, "score": h.score, "image_path": h.image_path}
                for h in _retrieve(query, top_k=top_k, doc_filter=doc_filter)
            ]
        },
        category="retrieval",
    ))

    register(ToolSpec(
        name="read_page_with_vlm",
        description="Read a single page image with a VLM and extract structured information per the instruction. Returns the extracted text. Use after retrieve_pages to read the content of retrieved pages.",
        params_schema=_ReadPageInput,
        output_schema=_ReadPageOutput,
        callable=lambda image_path, instruction: {"extracted_text": _vlm_read(image_path, instruction)},
        category="reading",
    ))

    register(ToolSpec(
        name="calculate",
        description="Safely evaluate an arithmetic expression. Supports + - * / ** // % and parentheses. Input must be a pure numeric expression (no Chinese characters, no variable names). Example: '(1500.5 - 800.2) / 1500.5 * 100'",
        params_schema=_CalculateInput,
        output_schema=_CalculateOutput,
        callable=lambda expression: {"value": _calc(expression)},
        category="compute",
    ))

    register(ToolSpec(
        name="disambiguate_caliber",
        description="Resolve cross-page numerical conflicts by extracting accounting caliber metadata (consolidated vs parent-company, FY vs CY). Pass the conflicting fact texts and the conflict topic.",
        params_schema=_DisambiguateInput,
        output_schema=_DisambiguateOutput,
        callable=lambda conflict_topic, fact_texts: _disambig_from_texts(conflict_topic, fact_texts),
        category="resolution",
    ))


def _disambig_from_texts(conflict_topic: str, fact_texts: list[str]) -> dict:
    """Shim: disambiguate_caliber expects Fact objects, but registry passes text strings."""
    from tools.disambiguate import disambiguate_caliber
    from agent.state import Fact
    facts = [Fact(text=t, source_doc="unknown", source_page=0) for t in fact_texts]
    return disambiguate_caliber(facts, conflict_topic)


_register_builtins()
