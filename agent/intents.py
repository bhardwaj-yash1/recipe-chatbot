"""
Intent definitions and registry for the Recipe Agent.

Defines 17 intents across 4 domains.  Each intent has an IntentConfig
specifying required/optional parameters, the execution engine method,
and response template.  The LLM's only job is to classify into one of
these intents and extract parameters — all computation is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Intent(str, Enum):
    """All 17 recognised intents."""

    # ── Search (5) ──
    SEARCH_BY_NAME = "SEARCH_BY_NAME"
    SEARCH_BY_INGREDIENTS = "SEARCH_BY_INGREDIENTS"
    SEARCH_SEMANTIC = "SEARCH_SEMANTIC"
    RECIPE_DETAIL = "RECIPE_DETAIL"
    REFINE_SEARCH = "REFINE_SEARCH"

    # ── Analytics (5) ──
    COUNT_RECIPES = "COUNT_RECIPES"
    INGREDIENT_FREQUENCY = "INGREDIENT_FREQUENCY"
    INGREDIENT_STATS = "INGREDIENT_STATS"
    RANK_BY_STEPS = "RANK_BY_STEPS"
    PERCENTAGE_CALC = "PERCENTAGE_CALC"

    # ── Matching (2) ──
    MATCH_BY_PANTRY = "MATCH_BY_PANTRY"
    REFINE_MATCH = "REFINE_MATCH"

    # ── Substitution (2) ──
    SUBSTITUTE_INGREDIENT = "SUBSTITUTE_INGREDIENT"
    ADAPT_RECIPE = "ADAPT_RECIPE"

    # ── General (3) ──
    GREETING = "GREETING"
    HELP = "HELP"
    UNCLEAR = "UNCLEAR"


@dataclass
class ParamSpec:
    """Specification for a single intent parameter."""
    type: str  # "string", "list", "int", "bool"
    description: str
    required: bool = True
    default: Any = None
    missing_prompt: str = ""  # Question to ask if required but missing


@dataclass
class IntentConfig:
    """Full configuration for a single intent."""
    intent: Intent
    description: str
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    executor: str = ""  # dot-path to engine method
    response_template: str = "default"
    domain: str = ""


# ---------------------------------------------------------------------------
# REGISTRY — the single source of truth for the agent's capabilities
# ---------------------------------------------------------------------------
INTENT_REGISTRY: Dict[Intent, IntentConfig] = {

    # ── SEARCH ────────────────────────────────────────────────────────
    Intent.SEARCH_BY_NAME: IntentConfig(
        intent=Intent.SEARCH_BY_NAME,
        description="Search for recipes by dish name or title keyword",
        domain="search",
        params={
            "query": ParamSpec("string", "Dish name or keyword to search for", required=True,
                               missing_prompt="What dish are you looking for?"),
            "limit": ParamSpec("int", "Maximum results to return", required=False, default=5),
        },
        executor="search_engine.search_by_name",
        response_template="recipe_list",
    ),
    Intent.SEARCH_BY_INGREDIENTS: IntentConfig(
        intent=Intent.SEARCH_BY_INGREDIENTS,
        description="Search for recipes containing specific ingredients",
        domain="search",
        params={
            "ingredients": ParamSpec("list", "List of ingredient names", required=True,
                                     missing_prompt="Which ingredients should the recipes contain?"),
            "mode": ParamSpec("string", "'all' for AND, 'any' for OR matching", required=False, default="all"),
            "limit": ParamSpec("int", "Maximum results", required=False, default=10),
        },
        executor="search_engine.search_by_ingredients",
        response_template="recipe_list",
    ),
    Intent.SEARCH_SEMANTIC: IntentConfig(
        intent=Intent.SEARCH_SEMANTIC,
        description="Natural language recipe search (e.g. 'something light for summer')",
        domain="search",
        params={
            "query": ParamSpec("string", "Natural language search query", required=True,
                               missing_prompt="What kind of recipe are you looking for?"),
            "limit": ParamSpec("int", "Maximum results", required=False, default=5),
        },
        executor="search_engine.search_semantic",
        response_template="recipe_list",
    ),
    Intent.RECIPE_DETAIL: IntentConfig(
        intent=Intent.RECIPE_DETAIL,
        description="Get full details of a specific recipe by ID or title",
        domain="search",
        params={
            "recipe_id": ParamSpec("string", "Stable recipe ID (e.g. ar_00001)", required=False),
            "title": ParamSpec("string", "Recipe title to look up", required=False),
        },
        executor="search_engine.get_recipe",
        response_template="recipe_detail",
    ),
    Intent.REFINE_SEARCH: IntentConfig(
        intent=Intent.REFINE_SEARCH,
        description="Refine/filter previous search results with new constraints",
        domain="search",
        params={
            "exclude_ingredients": ParamSpec("list", "Ingredients to exclude", required=False),
            "include_ingredients": ParamSpec("list", "Ingredients that must be present", required=False),
            "constraint": ParamSpec("string", "General constraint description", required=False),
        },
        executor="search_engine.refine_results",
        response_template="recipe_list",
    ),

    # ── ANALYTICS ─────────────────────────────────────────────────────
    Intent.COUNT_RECIPES: IntentConfig(
        intent=Intent.COUNT_RECIPES,
        description="Count how many recipes match given criteria",
        domain="analytics",
        params={
            "must_include": ParamSpec("list", "Ingredients that must be present", required=False),
            "must_exclude": ParamSpec("list", "Ingredients that must NOT be present", required=False),
            "title_keyword": ParamSpec("string", "Keyword in recipe title", required=False),
        },
        executor="analytics_engine.count_recipes",
        response_template="statistics",
    ),
    Intent.INGREDIENT_FREQUENCY: IntentConfig(
        intent=Intent.INGREDIENT_FREQUENCY,
        description="Find the most or least common ingredients",
        domain="analytics",
        params={
            "top_n": ParamSpec("int", "How many to return", required=False, default=10),
            "must_include": ParamSpec("list", "Filter to recipes containing these", required=False),
            "title_keyword": ParamSpec("string", "Filter by title keyword", required=False),
        },
        executor="analytics_engine.ingredient_frequency",
        response_template="statistics",
    ),
    Intent.INGREDIENT_STATS: IntentConfig(
        intent=Intent.INGREDIENT_STATS,
        description="Get mean/median/min/max ingredient counts",
        domain="analytics",
        params={
            "must_include": ParamSpec("list", "Filter to recipes containing these", required=False),
            "must_exclude": ParamSpec("list", "Exclude recipes with these", required=False),
            "title_keyword": ParamSpec("string", "Filter by title keyword", required=False),
        },
        executor="analytics_engine.ingredient_stats",
        response_template="statistics",
    ),
    Intent.RANK_BY_STEPS: IntentConfig(
        intent=Intent.RANK_BY_STEPS,
        description="Rank recipes by number of instruction steps",
        domain="analytics",
        params={
            "must_include": ParamSpec("list", "Filter to recipes containing these", required=False),
            "must_exclude": ParamSpec("list", "Exclude recipes with these", required=False),
            "title_keyword": ParamSpec("string", "Filter by title keyword", required=False),
            "ascending": ParamSpec("bool", "True for fewest-first, False for most-first", required=False, default=True),
            "limit": ParamSpec("int", "How many to return", required=False, default=5),
        },
        executor="analytics_engine.rank_by_steps",
        response_template="ranked_list",
    ),
    Intent.PERCENTAGE_CALC: IntentConfig(
        intent=Intent.PERCENTAGE_CALC,
        description="Calculate what percentage of recipes match criteria",
        domain="analytics",
        params={
            "must_include": ParamSpec("list", "Ingredients that must be present", required=False),
            "must_exclude": ParamSpec("list", "Ingredients that must NOT be present", required=False),
            "title_keyword": ParamSpec("string", "Filter by title keyword", required=False),
        },
        executor="analytics_engine.percentage",
        response_template="statistics",
    ),

    # ── MATCHING ──────────────────────────────────────────────────────
    Intent.MATCH_BY_PANTRY: IntentConfig(
        intent=Intent.MATCH_BY_PANTRY,
        description="Match recipes to available ingredients with constraints",
        domain="matching",
        params={
            "available": ParamSpec("list", "Ingredients the user has", required=True,
                                   missing_prompt="What ingredients do you have available?"),
            "excluded": ParamSpec("list", "Ingredients or groups to absolutely exclude", required=False),
            "max_steps": ParamSpec("int", "Maximum instruction steps allowed", required=False),
            "prefer": ParamSpec("string", "Ranking preference: fewest_missing, most_matched, fewest_total",
                                required=False, default="fewest_missing"),
            "limit": ParamSpec("int", "Maximum results", required=False, default=5),
        },
        executor="matching_engine.match_recipes",
        response_template="match_list",
    ),
    Intent.REFINE_MATCH: IntentConfig(
        intent=Intent.REFINE_MATCH,
        description="Update previous match with new constraints",
        domain="matching",
        params={
            "add_exclude": ParamSpec("list", "Additional ingredients to exclude", required=False),
            "remove_ingredient": ParamSpec("list", "Remove from available list", required=False),
            "add_available": ParamSpec("list", "Add to available list", required=False),
            "max_steps": ParamSpec("int", "New max steps constraint", required=False),
        },
        executor="matching_engine.match_recipes",
        response_template="match_list",
    ),

    # ── SUBSTITUTION ──────────────────────────────────────────────────
    Intent.SUBSTITUTE_INGREDIENT: IntentConfig(
        intent=Intent.SUBSTITUTE_INGREDIENT,
        description="Propose substitution for an ingredient in a specific recipe",
        domain="substitution",
        params={
            "recipe_id": ParamSpec("string", "Recipe ID", required=False),
            "recipe_title": ParamSpec("string", "Recipe title to search for", required=False),
            "ingredient_to_replace": ParamSpec("string", "The ingredient to replace", required=True,
                                                missing_prompt="Which ingredient do you want to replace?"),
            "restriction": ParamSpec("string", "Dietary restriction (e.g. dairy, eggs, nuts)", required=False),
        },
        executor="substitution_engine.find_substitution",
        response_template="substitution",
    ),
    Intent.ADAPT_RECIPE: IntentConfig(
        intent=Intent.ADAPT_RECIPE,
        description="Adapt an entire recipe for dietary restrictions",
        domain="substitution",
        params={
            "recipe_id": ParamSpec("string", "Recipe ID", required=False),
            "recipe_title": ParamSpec("string", "Recipe title to search for", required=False),
            "restrictions": ParamSpec("list", "Dietary restrictions to apply", required=True,
                                      missing_prompt="What dietary restrictions should be applied?"),
        },
        executor="substitution_engine.adapt_recipe",
        response_template="substitution",
    ),

    # ── GENERAL ───────────────────────────────────────────────────────
    Intent.GREETING: IntentConfig(
        intent=Intent.GREETING,
        description="User greeting or casual conversation",
        domain="general",
    ),
    Intent.HELP: IntentConfig(
        intent=Intent.HELP,
        description="User asking what the bot can do",
        domain="general",
    ),
    Intent.UNCLEAR: IntentConfig(
        intent=Intent.UNCLEAR,
        description="Ambiguous or unsupported query",
        domain="general",
    ),
}


def get_intent_descriptions() -> str:
    """Format all intents for injection into the classification prompt."""
    lines = []
    for intent, config in INTENT_REGISTRY.items():
        param_desc = ""
        if config.params:
            params_list = []
            for pname, pspec in config.params.items():
                req = "REQUIRED" if pspec.required else "optional"
                params_list.append(f"{pname} ({pspec.type}, {req}): {pspec.description}")
            param_desc = " | Params: " + "; ".join(params_list)
        lines.append(f"- {intent.value}: {config.description}{param_desc}")
    return "\n".join(lines)
