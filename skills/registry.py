"""Skill Registry (P32) — named bundles of prompt + tools + strategy.

A skill is a configuration-driven packaging of a prompt variant, tool list,
strategy defaults, and output schema. Skills are matched by trigger keywords
(first, O(1)) or semantic similarity (fallback, via memory.py cosine).

The planner calls match_skill(query) before generating a plan. If a skill
matches, it overrides default strategy (top_k, parallel, caliber_check, etc.)
and selects the correct prompt variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

_HERE = Path(__file__).resolve().parent


@dataclass
class SkillSpec:
    """A named reusable capability unit."""
    name: str                          # "cross_doc_compare"
    description: str = ""              # human-readable
    triggers: list[str] = field(default_factory=list)  # keyword triggers
    trigger_patterns: list[re.Pattern] = field(default_factory=list)  # compiled regex
    plan_template: str = "base"        # prompt variant name (maps to P30)
    tools: list[str] = field(default_factory=list)
    strategy: dict = field(default_factory=dict)
    verifier: str = "base"             # verifier variant
    output_schema: str = "text"


SKILLS: dict[str, SkillSpec] = {}


def _compile_triggers(triggers: list[str]) -> list[re.Pattern]:
    """Compile trigger strings to regex patterns. Supports basic regex in triggers."""
    patterns = []
    for t in triggers:
        try:
            patterns.append(re.compile(t))
        except re.error:
            patterns.append(re.compile(re.escape(t)))
    return patterns


def load_skills() -> None:
    """Load all YAML skill definitions from skills/ directory."""
    global SKILLS
    if SKILLS:
        return

    for yaml_path in sorted(_HERE.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not data or "skill" not in data:
                continue
            name = data["skill"]
            triggers = data.get("triggers") or []
            SKILLS[name] = SkillSpec(
                name=name,
                description=data.get("description", ""),
                triggers=triggers,
                trigger_patterns=_compile_triggers(triggers),
                plan_template=data.get("plan_template", "base"),
                tools=data.get("tools", []),
                strategy=data.get("strategy", {}),
                verifier=data.get("verifier", "base"),
                output_schema=data.get("output_schema", "text"),
            )
            logger.debug(f"loaded skill: {name}")
        except Exception as e:
            logger.warning(f"failed to load skill from {yaml_path.name}: {e}")

    logger.info(f"Loaded {len(SKILLS)} skills: {list(SKILLS.keys())}")


def match_skill(query: str) -> Optional[SkillSpec]:
    """Match a user query to the best-fit skill.

    Phase 1: Trigger keyword match (O(1) — most queries hit here).
    Phase 2: Fallback to semantic match (via memory.py cosine) if no keyword hit.

    Returns the best matching SkillSpec, or None if no match (use generic/default).
    """
    load_skills()

    if not query or not SKILLS:
        return None

    # Phase 1: keyword trigger match
    best: Optional[SkillSpec] = None
    best_score = 0

    for name, spec in SKILLS.items():
        score = 0
        for pat in spec.trigger_patterns:
            matches = pat.findall(query)
            score += len(matches)
        if score > best_score:
            best_score = score
            best = spec

    if best is not None and best_score > 0:
        logger.info(f"skill match: {best.name} (keyword score={best_score})")
        return best

    # Phase 2: semantic fallback (deferred to P28 memory system — for now, return None)
    if best_score == 0:
        logger.debug("skill match: no keyword hit, using default planner")
        return None

    return best


def get_skill_strategy(query: str) -> dict:
    """Get merged strategy defaults from the matched skill for this query.

    Returns a dict with {retrieve_top_k, parallel_fan_out, caliber_check, ...}
    that the executor and planner can consume. Empty dict = use defaults.
    """
    skill = match_skill(query)
    if skill is None:
        return {}
    return dict(skill.strategy)


def get_skill_tools(query: str) -> list[str]:
    """Get the recommended tool list for a query's matched skill."""
    skill = match_skill(query)
    if skill is None:
        return ["retrieve_pages", "read_page_with_vlm"]  # default
    return list(skill.tools)


def get_skill_verifier_variant(query: str) -> str:
    """Get the recommended verifier variant for a query's matched skill."""
    skill = match_skill(query)
    if skill is None:
        return "base"
    return skill.verifier


# Auto-load on import
load_skills()
