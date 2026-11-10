"""
Matching Engine — ingredient-based recipe matching with explainable scoring.

Given a list of available ingredients and constraints, retrieves and ranks
the closest existing recipes from the dataset.  **Never invents new recipes.**
All operations are fully deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from core.models import Recipe
from core.normalizer import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dairy / common allergen ingredient groups for constraint checking
# ---------------------------------------------------------------------------
INGREDIENT_GROUPS: Dict[str, Set[str]] = {
    "dairy": {
        "milk", "cream", "butter", "cheese", "yogurt", "sour cream",
        "heavy cream", "whipping cream", "half-and-half", "buttermilk",
        "cream cheese", "ricotta", "mozzarella", "parmesan", "cheddar",
        "gruyere", "mascarpone", "brie", "feta", "gouda", "provolone",
        "cottage cheese", "evaporated milk", "condensed milk", "ghee",
        "ice cream", "whey",
    },
    "nuts": {
        "peanut", "almond", "walnut", "cashew", "pecan", "pistachio",
        "macadamia", "hazelnut", "brazil nut", "pine nut", "chestnut",
        "peanut butter", "almond butter", "almond milk",
    },
    "gluten": {
        "flour", "all-purpose flour", "bread flour", "wheat", "pasta",
        "spaghetti", "penne", "noodle", "bread", "breadcrumb",
        "panko", "tortilla", "couscous", "barley", "rye", "semolina",
    },
    "eggs": {"egg", "eggs", "egg white", "egg yolk"},
    "shellfish": {
        "shrimp", "crab", "lobster", "clam", "mussel", "oyster",
        "scallop", "crawfish", "prawn",
    },
    "soy": {"soy sauce", "soy", "tofu", "tempeh", "edamame", "miso"},
}


def _expand_exclusions(excluded: List[str]) -> Set[str]:
    """Expand ingredient names and group names into a flat set."""
    expanded: Set[str] = set()
    for item in excluded:
        low = item.lower().strip()
        if low in INGREDIENT_GROUPS:
            expanded.update(INGREDIENT_GROUPS[low])
        else:
            expanded.add(normalize(low, strip_noise=True))
    return expanded


@dataclass
class MatchResult:
    """A single recipe match with full explainability."""
    recipe: Recipe
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    extra_ingredients: List[str]
    score: float
    step_count: int
    rank_explanation: str
    violates_constraint: bool = False
    violation_details: str = ""


class MatchingEngine:
    """Deterministic ingredient-based recipe matching with constraint enforcement."""

    def __init__(self, recipes: List[Recipe]):
        self._recipes = recipes

    def match_recipes(
        self,
        available: List[str],
        excluded: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        prefer: str = "fewest_missing",
        limit: int = 5,
    ) -> List[MatchResult]:
        """
        Match recipes to available ingredients with hard constraints.

        Parameters
        ----------
        available : list[str]
            Ingredients the user has.
        excluded : list[str], optional
            Ingredients or groups to absolutely exclude (hard constraint).
        max_steps : int, optional
            Maximum instruction steps allowed (hard constraint).
        prefer : str
            Ranking preference: 'fewest_missing', 'most_matched', 'fewest_total'.
        limit : int
            Maximum results.

        Returns
        -------
        list[MatchResult]
            Ranked matches. Empty list if no eligible recipe exists.
        """
        norm_available: Set[str] = {normalize(a, strip_noise=True) for a in available}
        excl_set = _expand_exclusions(excluded or [])

        results: List[MatchResult] = []

        for recipe in self._recipes:
            recipe_ings = set(recipe.ingredient_names)

            # ── HARD CONSTRAINT: excluded ingredients ──
            if excl_set:
                violations = recipe_ings & excl_set
                if not violations:
                    for ri in recipe_ings:
                        for ex in excl_set:
                            if ex in ri or (len(ri) >= 3 and ri in ex):
                                violations.add(ri)
                                break
                        if violations:
                            break

                if violations:
                    continue  # NEVER silently relax

            # ── HARD CONSTRAINT: max steps ──
            step_count = len(recipe.instruction_steps)
            if max_steps is not None and step_count > max_steps:
                continue

            # ── Scoring ──
            matched = recipe_ings & norm_available
            # Also partial ingredient name matching
            for ri in recipe_ings - matched:
                for av in norm_available:
                    if av in ri or ri in av:
                        matched.add(ri)
                        break

            missing = recipe_ings - matched
            extra = norm_available - recipe_ings

            if not recipe_ings:
                continue

            coverage = len(matched) / len(recipe_ings)

            # Score variants
            if prefer == "fewest_missing":
                score = coverage - (len(missing) * 0.1)
            elif prefer == "most_matched":
                score = len(matched)
            elif prefer == "fewest_total":
                score = coverage - (len(recipe_ings) * 0.01)
            else:
                score = coverage

            explanation = (
                f"Coverage: {len(matched)}/{len(recipe_ings)} ingredients matched "
                f"({coverage:.0%}). "
                f"Missing {len(missing)} ingredient(s). "
                f"Ranking by: {prefer}."
            )

            results.append(MatchResult(
                recipe=recipe,
                matched_ingredients=sorted(matched),
                missing_ingredients=sorted(missing),
                extra_ingredients=sorted(extra - recipe_ings),
                score=score,
                step_count=step_count,
                rank_explanation=explanation,
            ))

        # Sort by score descending
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:limit]

    def closest_with_violations(
        self,
        available: List[str],
        excluded: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        limit: int = 3,
    ) -> List[MatchResult]:
        """
        Find closest recipes that *violate* a constraint.

        These are returned separately and clearly marked — never mixed
        with valid results.
        """
        norm_available: Set[str] = {normalize(a, strip_noise=True) for a in available}
        excl_set = _expand_exclusions(excluded or [])

        results: List[MatchResult] = []

        for recipe in self._recipes:
            recipe_ings = set(recipe.ingredient_names)
            step_count = len(recipe.instruction_steps)

            violations = set()
            for ri in recipe_ings:
                for ex in excl_set:
                    if ex in ri or ri in ex:
                        violations.add(ri)

            steps_violated = max_steps is not None and step_count > max_steps

            if not violations and not steps_violated:
                continue  # This recipe is valid — not what we want

            matched = set()
            for ri in recipe_ings:
                for av in norm_available:
                    if av in ri or ri in av:
                        matched.add(ri)
                        break

            missing = recipe_ings - matched
            coverage = len(matched) / len(recipe_ings) if recipe_ings else 0
            score = coverage

            details = []
            if violations:
                details.append(f"Contains excluded: {', '.join(sorted(violations))}")
            if steps_violated:
                details.append(f"Has {step_count} steps (limit: {max_steps})")

            results.append(MatchResult(
                recipe=recipe,
                matched_ingredients=sorted(matched),
                missing_ingredients=sorted(missing),
                extra_ingredients=[],
                score=score,
                step_count=step_count,
                rank_explanation=f"Coverage: {len(matched)}/{len(recipe_ings)} ({coverage:.0%})",
                violates_constraint=True,
                violation_details="; ".join(details),
            ))

        results.sort(key=lambda m: m.score, reverse=True)
        return results[:limit]
