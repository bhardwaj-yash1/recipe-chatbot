"""
Orchestrator — the central Deterministic State Machine controller.

Inspired by the reference project's architecture, this orchestrator
ensures the LLM is NEVER used for data computation.  The flow is:

    IDLE → CLASSIFYING → EXTRACTING → [CLARIFYING] → EXECUTING → RESPONDING → IDLE

The LLM is invoked only for:
  1. Intent classification (CLASSIFYING)
  2. Parameter extraction (EXTRACTING)
  3. Response generation (RESPONDING)

Everything else — search, counting, ranking, matching, substitution —
is done by deterministic code in the engine layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent.context import ConversationContext, MatchContext, Turn
from agent.hallucination_guard import scrub_hallucinated_params
from agent.intents import Intent, IntentConfig, INTENT_REGISTRY, ParamSpec
from agent.llm_client import LLMClient
from core.models import Recipe
from engine.analytics_engine import AnalyticsEngine, AnalyticsResult
from engine.matching_engine import MatchingEngine, MatchResult
from engine.search_engine import SearchEngine
from engine.substitution_engine import SubstitutionEngine, SubstitutionResult
from prompts.templates import build_classify_prompt, build_extract_prompt, build_respond_prompt, build_no_results_prompt

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    CLASSIFYING = "CLASSIFYING"
    EXTRACTING = "EXTRACTING"
    CLARIFYING = "CLARIFYING"
    EXECUTING = "EXECUTING"
    RESPONDING = "RESPONDING"


# Dietary-constraint keyword mapping for REFINE_SEARCH
_DIETARY_KEYWORDS: Dict[str, List[str]] = {
    "dairy": ["milk", "cream", "butter", "cheese", "yogurt", "sour cream",
              "whipping cream", "buttermilk", "cream cheese", "ricotta",
              "mozzarella", "parmesan", "cheddar", "ghee", "ice cream"],
    "vegan": ["milk", "cream", "butter", "cheese", "yogurt", "egg",
              "honey", "gelatin", "meat", "chicken", "beef", "pork",
              "fish", "shrimp"],
    "vegetarian": ["chicken", "beef", "pork", "lamb", "turkey", "fish",
                   "shrimp", "bacon", "sausage", "meat"],
    "gluten": ["flour", "bread", "pasta", "noodle", "cracker", "breadcrumb",
               "wheat", "barley", "rye", "couscous"],
    "nut": ["peanut", "almond", "walnut", "cashew", "pecan", "pistachio",
            "hazelnut", "pine nut", "macadamia"],
}


_SOURCE_NAMES: Dict[str, str] = {
    "ar": "AllRecipes",
    "epi": "Epicurious",
    "fn": "Food Network",
}


def get_source_name(source: str) -> str:
    """Map source code to human-readable publication name."""
    return _SOURCE_NAMES.get(source.lower(), source.title())


def _get_source_name(source: str) -> str:
    return get_source_name(source)


class Orchestrator:
    """
    Central state machine that routes user messages through the
    deterministic classification → extraction → execution pipeline.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        analytics_engine: AnalyticsEngine,
        matching_engine: MatchingEngine,
        substitution_engine: SubstitutionEngine,
        llm_client: LLMClient,
    ):
        self.search = search_engine
        self.analytics = analytics_engine
        self.matching = matching_engine
        self.substitution = substitution_engine
        self.llm = llm_client
        self.context = ConversationContext()
        self._state = State.IDLE

    def reset(self):
        """Reset the conversation state."""
        self.context = ConversationContext()
        self._state = State.IDLE

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def handle_message(self, user_message: str) -> str:
        """
        Process a user message through the full state machine pipeline.

        Returns the bot's natural language response.
        """
        logger.info("─── New message: %s", user_message[:100])

        # ── STEP 1: CLASSIFYING ──────────────────────────────────────
        self._state = State.CLASSIFYING
        intent, params, confidence = self._classify(user_message)
        logger.info("  Intent: %s (confidence=%.2f)", intent.value, confidence)

        # ── STEP 2: EXTRACTING (+ hallucination scrub) ───────────────
        self._state = State.EXTRACTING
        config = INTENT_REGISTRY.get(intent)
        if config and config.params:
            params = self._extract_and_scrub(user_message, intent, config, params)
            logger.info("  Params after scrub: %s", params)

        # ── STEP 3: Check for missing required params → CLARIFYING ───
        if config and config.params:
            missing = self._find_missing_required(config, params)
            if missing:
                self._state = State.CLARIFYING
                clarification = self._build_clarification(missing, config)
                turn = Turn(user_message=user_message, intent=intent, params=params, response=clarification)
                self.context.add_turn(turn)
                self._state = State.IDLE
                return clarification

        # ── STEP 4: EXECUTING ────────────────────────────────────────
        self._state = State.EXECUTING
        result_data, result_text = self._execute(intent, params, user_message)

        # ── STEP 5: RESPONDING ───────────────────────────────────────
        self._state = State.RESPONDING
        if result_text:
            response = self._generate_response(result_text, user_message)
        else:
            response = self._handle_general(intent, user_message)

        # ── Record turn and reset ────────────────────────────────────
        turn = Turn(
            user_message=user_message,
            intent=intent,
            params=params,
            response=response,
            result_data=result_data,
        )
        self.context.add_turn(turn)
        self._state = State.IDLE

        return response

    # ==================================================================
    # CLASSIFICATION
    # ==================================================================
    def _classify(self, user_message: str) -> Tuple[Intent, Dict, float]:
        """Classify user message into an intent using the LLM."""
        prompt = build_classify_prompt(self.context.get_context_summary())
        result = self.llm.invoke_json(prompt, user_message)

        intent_str = result.get("intent", "UNCLEAR")
        confidence = float(result.get("confidence", 0.5))
        params = result.get("params", {})

        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.UNCLEAR

        return intent, params, confidence

    # ==================================================================
    # EXTRACTION + HALLUCINATION GUARD
    # ==================================================================
    def _extract_and_scrub(
        self,
        user_message: str,
        intent: Intent,
        config: IntentConfig,
        initial_params: Dict,
    ) -> Dict:
        """Extract parameters and scrub hallucinated values."""
        # If initial classification already extracted params, use those
        params = initial_params or {}

        # Check if we need deeper extraction
        missing = self._find_missing_required(config, params)
        if missing:
            # Do a focused extraction
            param_specs = "\n".join(
                f"- {name} ({spec.type}): {spec.description}"
                for name, spec in config.params.items()
            )
            prompt = build_extract_prompt(
                intent.value,
                param_specs,
                self.context.get_context_summary(),
            )
            extracted = self.llm.invoke_json(prompt, user_message)
            params.update(extracted)

        # Apply hallucination guard
        params = scrub_hallucinated_params(
            params,
            user_message,
            self.context.get_user_messages(),
            intent.value,
        )

        return params

    def _find_missing_required(self, config: IntentConfig, params: Dict) -> List[str]:
        """Find required parameters that are missing."""
        missing = []
        for name, spec in config.params.items():
            if spec.required and name not in params:
                missing.append(name)
        return missing

    def _build_clarification(self, missing: List[str], config: IntentConfig) -> str:
        """Build a clarification question for missing parameters."""
        questions = []
        for name in missing:
            spec = config.params.get(name)
            if spec and spec.missing_prompt:
                questions.append(spec.missing_prompt)
            else:
                questions.append(f"Could you please specify the {name.replace('_', ' ')}?")
        return " ".join(questions)

    # ==================================================================
    # EXECUTION — routes to deterministic engines
    # ==================================================================
    def _execute(self, intent: Intent, params: Dict, user_message: str) -> Tuple[Any, str]:
        """Execute the classified intent using deterministic engines."""

        # ── SEARCH ────────────────────────────────────────────────────
        if intent in (Intent.SEARCH_BY_NAME, Intent.SEARCH_SEMANTIC):
            raw_q = params.get("query", user_message)
            low_msg = user_message.lower()

            # Detect open-ended culinary inspiration requests
            open_ended = any(
                p in low_msg for p in [
                    "what should i cook", "what to cook", "what can i cook",
                    "suggest something", "recommend dinner", "recommend lunch",
                    "what do you recommend", "dinner idea", "meal idea", "give me an idea"
                ]
            )

            if open_ended:
                # Provide a diverse menu of 3 top-tier, complete recipes
                r_pasta = self.search.search_by_name("pasta", limit=1)
                r_chicken = self.search.search_by_name("chicken", limit=1)
                r_veg = self.search.search_by_name("salad", limit=1)
                recipes = [r for r in (r_pasta + r_chicken + r_veg) if r]
                if not recipes:
                    recipes = self.search.search_by_name("dinner", limit=3)
            elif intent == Intent.SEARCH_BY_NAME:
                recipes = self.search.search_by_name(raw_q, limit=int(params.get("limit", 4)))
                # If BM25 returned fewer than 2, try hybrid
                if len(recipes) < 2:
                    hybrid = self.search.hybrid_search(raw_q, limit=4)
                    for hr in hybrid:
                        if hr.recipe_id not in [r.recipe_id for r in recipes]:
                            recipes.append(hr)
            else:
                recipes = self.search.hybrid_search(raw_q, limit=int(params.get("limit", 4)))

            self.context.set_search_results([r.recipe_id for r in recipes])
            return recipes, self._format_recipe_list(recipes)

        if intent == Intent.SEARCH_BY_INGREDIENTS:
            recipes = self.search.search_by_ingredients(
                params.get("ingredients", []),
                mode=params.get("mode", "all"),
                limit=int(params.get("limit", 10)),
            )
            self.context.set_search_results([r.recipe_id for r in recipes])
            return recipes, self._format_recipe_list(recipes)

        if intent == Intent.RECIPE_DETAIL:
            recipe = None
            if params.get("recipe_id"):
                recipe = self.search.get_recipe(params["recipe_id"])
            elif params.get("title"):
                results = self.search.search_by_name(params["title"], limit=1)
                recipe = results[0] if results else None
            if recipe:
                self.context.set_last_recipe(recipe.recipe_id)
                return recipe, self._format_recipe_detail(recipe)
            return None, ""

        if intent == Intent.REFINE_SEARCH:
            if not self.context.active_search_results:
                return None, "No active search to refine. Please search for recipes first."

            exclude = params.get("exclude_ingredients", [])
            include = params.get("include_ingredients", [])
            constraint = params.get("constraint", "")

            # Expand dietary constraints
            if constraint:
                constraint_lower = constraint.lower()
                for diet_key, diet_ings in _DIETARY_KEYWORDS.items():
                    if diet_key in constraint_lower:
                        exclude.extend(diet_ings)
                        break
                # Also handle "no X" patterns
                if not exclude:
                    exclude = [constraint.replace("no ", "").replace("without ", "").strip()]

            recipes = self.search.refine_results(
                self.context.active_search_results,
                exclude_ingredients=exclude if exclude else None,
                include_ingredients=include if include else None,
            )
            self.context.set_search_results([r.recipe_id for r in recipes])
            self.context.add_filter("exclude_ingredients", exclude)
            return recipes, self._format_recipe_list(recipes, note=f"Refined from {len(self.context.active_search_results)} results")

        # ── ANALYTICS ─────────────────────────────────────────────────
        if intent == Intent.COUNT_RECIPES:
            result = self.analytics.count_recipes(
                must_include=params.get("must_include"),
                must_exclude=params.get("must_exclude"),
                title_keyword=params.get("title_keyword"),
            )
            return result, self._format_analytics(result)

        if intent == Intent.PERCENTAGE_CALC:
            result = self.analytics.percentage(
                must_include=params.get("must_include"),
                must_exclude=params.get("must_exclude"),
                title_keyword=params.get("title_keyword"),
            )
            return result, self._format_analytics(result)

        if intent == Intent.INGREDIENT_FREQUENCY:
            result = self.analytics.ingredient_frequency(
                top_n=int(params.get("top_n", 10)),
                must_include=params.get("must_include"),
                title_keyword=params.get("title_keyword"),
            )
            return result, self._format_analytics(result)

        if intent == Intent.INGREDIENT_STATS:
            result = self.analytics.ingredient_stats(
                must_include=params.get("must_include"),
                must_exclude=params.get("must_exclude"),
                title_keyword=params.get("title_keyword"),
            )
            return result, self._format_analytics(result)

        if intent == Intent.RANK_BY_STEPS:
            result = self.analytics.rank_by_steps(
                must_include=params.get("must_include"),
                must_exclude=params.get("must_exclude"),
                title_keyword=params.get("title_keyword"),
                ascending=params.get("ascending", True),
                limit=int(params.get("limit", 5)),
            )
            return result, self._format_analytics(result)

        # ── MATCHING ──────────────────────────────────────────────────
        if intent == Intent.MATCH_BY_PANTRY:
            available = params.get("available", [])
            excluded = params.get("excluded", [])
            max_steps = params.get("max_steps")
            if max_steps is not None:
                max_steps = int(max_steps)
            prefer = params.get("prefer", "fewest_missing")

            matches = self.matching.match_recipes(
                available=available,
                excluded=excluded,
                max_steps=max_steps,
                prefer=prefer,
                limit=int(params.get("limit", 5)),
            )

            # Save match context for refinement
            self.context.set_match_context(MatchContext(
                available=available,
                excluded=excluded,
                max_steps=max_steps,
                prefer=prefer,
            ))

            if not matches:
                # Try closest with violations
                closest = self.matching.closest_with_violations(
                    available=available, excluded=excluded, max_steps=max_steps, limit=3,
                )
                return (matches, closest), self._format_no_match(closest)

            return matches, self._format_match_results(matches)

        if intent == Intent.REFINE_MATCH:
            if not self.context.active_match_context:
                return None, "No active matching session. Please start with your available ingredients."

            mc = self.context.active_match_context
            if params.get("add_exclude"):
                mc.excluded.extend(params["add_exclude"])
            if params.get("remove_ingredient"):
                for rm in params["remove_ingredient"]:
                    if rm in mc.available:
                        mc.available.remove(rm)
            if params.get("add_available"):
                mc.available.extend(params["add_available"])
            if params.get("max_steps") is not None:
                mc.max_steps = int(params["max_steps"])

            matches = self.matching.match_recipes(
                available=mc.available,
                excluded=mc.excluded,
                max_steps=mc.max_steps,
                prefer=mc.prefer,
                limit=5,
            )
            self.context.set_match_context(mc)

            if not matches:
                closest = self.matching.closest_with_violations(
                    available=mc.available, excluded=mc.excluded, max_steps=mc.max_steps, limit=3,
                )
                return (matches, closest), self._format_no_match(closest)

            return matches, self._format_match_results(matches)

        # ── SUBSTITUTION ──────────────────────────────────────────────
        if intent == Intent.SUBSTITUTE_INGREDIENT:
            recipe = None
            if params.get("recipe_id"):
                recipe = self.search.get_recipe(params["recipe_id"])
            elif params.get("recipe_title"):
                results = self.search.search_by_name(params["recipe_title"], limit=1)
                recipe = results[0] if results else None
            elif self.context.last_recipe_id:
                recipe = self.search.get_recipe(self.context.last_recipe_id)

            if not recipe:
                return None, "I couldn't find the recipe. Please specify a recipe ID or title."

            ingredient = params.get("ingredient_to_replace", "")
            if not ingredient:
                return None, "Which ingredient would you like to replace?"

            result = self.substitution.find_substitution(
                recipe=recipe,
                ingredient_to_replace=ingredient,
                restriction=params.get("restriction"),
            )
            return result, self._format_substitution(result)

        if intent == Intent.ADAPT_RECIPE:
            recipe = None
            if params.get("recipe_id"):
                recipe = self.search.get_recipe(params["recipe_id"])
            elif params.get("recipe_title"):
                results = self.search.search_by_name(params["recipe_title"], limit=1)
                recipe = results[0] if results else None
            elif self.context.last_recipe_id:
                recipe = self.search.get_recipe(self.context.last_recipe_id)

            if not recipe:
                return None, "I couldn't find the recipe. Please specify a recipe ID or title."

            restrictions = params.get("restrictions", [])
            all_results = []
            for restriction in restrictions:
                # Find ingredients in the recipe that match the restriction
                from engine.matching_engine import INGREDIENT_GROUPS, _expand_exclusions
                restricted_ings = _expand_exclusions([restriction])
                for ing_name in recipe.ingredient_names:
                    if any(ri in ing_name or ing_name in ri for ri in restricted_ings):
                        result = self.substitution.find_substitution(
                            recipe=recipe,
                            ingredient_to_replace=ing_name,
                            restriction=restriction,
                        )
                        all_results.append(result)

            if not all_results:
                return None, f"No ingredients in this recipe match the restriction(s): {restrictions}. No changes needed!"

            combined_text = "\n\n".join(self._format_substitution(r) for r in all_results)
            return all_results, combined_text

        # ── GENERAL ───────────────────────────────────────────────────
        return None, ""

    # ==================================================================
    # RESPONSE GENERATION
    # ==================================================================
    def _generate_response(self, result_text: str, user_message: str) -> str:
        """Use LLM to format results into a natural language response."""
        prompt = build_respond_prompt(result_text, user_message)
        return self.llm.invoke(prompt, user_message)

    def _handle_general(self, intent: Intent, user_message: str) -> str:
        """Handle general intents without engine execution."""
        if intent == Intent.GREETING:
            return (
                "👋 Hello! I'm the Recipe Agent — your conversational recipe assistant. "
                "I can search through 125,000+ recipes, compute statistics, match recipes "
                "to your pantry, and suggest ingredient substitutions. What would you like to do?"
            )
        if intent == Intent.HELP:
            return (
                "Here's what I can do:\n\n"
                "🔍 **Search**: Find recipes by name, ingredients, or natural language\n"
                "📊 **Analytics**: Count recipes, find common ingredients, rank by steps\n"
                "🧑‍🍳 **Matching**: Match recipes to ingredients you have on hand\n"
                "🔄 **Substitution**: Suggest ingredient replacements with evidence\n\n"
                "I support follow-up questions and will remember your search context. "
                "Try asking something like:\n"
                "- *Find a penne pasta recipe*\n"
                "- *How many recipes contain both garlic and onion?*\n"
                "- *I have eggs, onions, and rice — what can I make?*\n"
                "- *Replace eggs in this banana bread recipe*"
            )
        # UNCLEAR
        return (
            "I'm not sure I understood that. I can help with:\n"
            "• Searching recipes by name, ingredients, or description\n"
            "• Computing statistics (counts, percentages, rankings)\n"
            "• Matching recipes to your available ingredients\n"
            "• Suggesting ingredient substitutions\n\n"
            "Could you rephrase your question?"
        )

    # ==================================================================
    # FORMATTING HELPERS  (deterministic text → LLM for final polish)
    # ==================================================================
    def _format_recipe_list(self, recipes: List[Recipe], note: str = "") -> str:
        if not recipes:
            return "No recipes found matching your criteria."
        lines = []
        if note:
            lines.append(f"[{note}]\n")

        # Featured primary recipe in full detail
        primary = recipes[0]
        self.context.set_last_recipe(primary.recipe_id)
        src_name = _get_source_name(primary.source)

        lines.append(f"### Featured Recipe: **{primary.title}**")
        lines.append(f"Source: {src_name} | Instructions: {len(primary.instruction_steps)} steps | Ingredients: {len(primary.ingredients_raw)}\n")

        lines.append("**Ingredients:**")
        for ing in primary.ingredients_raw:
            lines.append(f"- {ing}")

        lines.append("\n**Cooking Instructions:**")
        for i, step in enumerate(primary.instruction_steps, 1):
            lines.append(f"{i}. {step}")

        # Additional options if available
        if len(recipes) > 1:
            lines.append("\n---\n### Other Delicious Options Found in the Dataset:")
            for i, r in enumerate(recipes[1:4], 2):
                r_src = _get_source_name(r.source)
                lines.append(f"{i}. **{r.title}** ({r_src}) — {len(r.instruction_steps)} step(s)")
                if r.ingredients_raw:
                    preview = ", ".join(r.ingredients_raw[:4])
                    lines.append(f"   *Key ingredients:* {preview}...")
                lines.append("")

        return "\n".join(lines)

    def _format_recipe_detail(self, recipe: Recipe) -> str:
        src_name = _get_source_name(recipe.source)
        lines = [
            f"**{recipe.title}** ({src_name})",
            f"Source: {src_name}",
            f"\n**Ingredients ({len(recipe.ingredients_raw)}):**",
        ]
        for ing in recipe.ingredients_raw:
            lines.append(f"  • {ing}")
        lines.append(f"\n**Instructions ({len(recipe.instruction_steps)} steps):**")
        for i, step in enumerate(recipe.instruction_steps, 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    def _format_analytics(self, result: AnalyticsResult) -> str:
        lines = [
            f"**{result.metric.replace('_', ' ').title()}**: {result.value}",
            f"Filters applied: {result.filters_applied}",
            f"Recipes analyzed: {result.total_analyzed:,} / {result.total_dataset:,} total",
            f"Method: {result.method}",
        ]
        if result.details:
            lines.append(f"Details: {json.dumps(result.details)}")
        return "\n".join(lines)

    def _format_match_results(self, matches: List[MatchResult]) -> str:
        lines = [f"Found {len(matches)} matching recipe(s):\n"]
        for i, m in enumerate(matches, 1):
            src_name = _get_source_name(m.recipe.source)
            lines.append(f"{i}. **{m.recipe.title}** ({src_name})")
            lines.append(f"   Score: {m.score:.2f} | Steps: {m.step_count}")
            lines.append(f"   ✅ Matched: {', '.join(m.matched_ingredients) if m.matched_ingredients else 'none'}")
            lines.append(f"   ❌ Missing: {', '.join(m.missing_ingredients) if m.missing_ingredients else 'none'}")
            lines.append(f"   📝 {m.rank_explanation}")
            lines.append("")
        lines.append("*Note: Ingredient matches are verified at name level; quantity sufficiency was not verified against raw package weights.*")
        return "\n".join(lines)

    def _format_no_match(self, closest: List[MatchResult]) -> str:
        lines = ["**No recipes found matching ALL your constraints.**\n"]
        if closest:
            lines.append("The closest alternatives (⚠️ these VIOLATE one or more constraints):\n")
            for i, m in enumerate(closest, 1):
                src_name = _get_source_name(m.recipe.source)
                lines.append(f"{i}. **{m.recipe.title}** ({src_name})")
                lines.append(f"   ⚠️ Violation: {m.violation_details}")
                lines.append(f"   Matched: {', '.join(m.matched_ingredients)}")
                lines.append("")
        else:
            lines.append("No close alternatives found either.")
        return "\n".join(lines)

    def _format_substitution(self, result: SubstitutionResult) -> str:
        lines = [
            f"**Substitution for '{result.ingredient_replaced}' in {result.original_recipe.title}**",
            f"Ingredient role(s): {', '.join(result.ingredient_roles)}",
            f"\n**Suggested substitutes:**",
        ]
        for i, c in enumerate(result.candidates[:5], 1):
            lines.append(f"  {i}. **{c.substitute_name}** (confidence: {c.confidence})")
            lines.append(f"     Evidence: {c.evidence_type} — {c.notes}")
            if c.supporting_recipe_ids:
                sup_titles = []
                for rid in c.supporting_recipe_ids[:3]:
                    rec = self.substitution._by_id.get(rid) if hasattr(self, 'substitution') else None
                    if rec:
                        sup_titles.append(f"'{rec.title}' ({_get_source_name(rec.source)})")
                    else:
                        sup_titles.append(_get_source_name(rid.split('_')[0]))
                lines.append(f"     Supporting dataset recipes: {', '.join(sup_titles)}")
        lines.append(f"\n**Modified ingredient list:**")
        for ing in result.modified_ingredients:
            lines.append(f"  • {ing}")
        lines.append(f"\n**Impact assessment:**")
        lines.append(f"  {result.impact_notes}")
        lines.append(f"\n**Evidence summary:** {result.evidence_summary}")
        if result.restriction:
            lines.append(f"Restriction applied: {result.restriction}")
        return "\n".join(lines)
