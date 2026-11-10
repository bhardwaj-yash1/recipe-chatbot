"""
Substitution Engine — evidence-based ingredient substitution.

Uses the local recipe dataset as the PRIMARY source of evidence.
Searches for similar recipes that already use alternative ingredients,
analyses ingredient roles, and produces modified recipes with cited evidence.

No LLM calls.  All analysis is deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.models import Recipe
from core.normalizer import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingredient role classification
# ---------------------------------------------------------------------------
ROLE_MAP: Dict[str, List[str]] = {
    "binding": ["egg", "flax", "chia seed", "gelatin", "aquafaba", "cornstarch", "arrowroot"],
    "leavening": ["baking soda", "baking powder", "yeast", "egg", "cream of tartar", "self-rising flour"],
    "fat": [
        "butter", "oil", "olive oil", "vegetable oil", "coconut oil", "shortening",
        "lard", "ghee", "margarine", "avocado oil", "canola oil",
    ],
    "moisture": [
        "milk", "water", "buttermilk", "cream", "yogurt", "sour cream",
        "applesauce", "banana", "pumpkin puree", "coconut milk",
    ],
    "sweetener": [
        "sugar", "brown sugar", "honey", "maple syrup", "agave", "molasses",
        "powdered sugar", "corn syrup", "stevia",
    ],
    "thickener": [
        "flour", "cornstarch", "arrowroot", "tapioca", "gelatin", "agar",
        "cream", "roux", "potato starch",
    ],
    "acid": [
        "lemon juice", "lime juice", "vinegar", "wine", "tomato", "buttermilk",
        "citric acid", "orange juice",
    ],
    "protein": [
        "chicken", "beef", "pork", "tofu", "tempeh", "egg", "fish",
        "shrimp", "turkey", "lamb", "seitan", "beans", "lentil",
    ],
    "flavor": [
        "vanilla", "cinnamon", "garlic", "onion", "salt", "pepper",
        "cumin", "paprika", "oregano", "basil", "thyme", "ginger",
        "nutmeg", "soy sauce", "worcestershire",
    ],
    "texture": [
        "breadcrumb", "panko", "nut", "coconut", "oat", "cracker",
        "crispy", "crunchy",
    ],
}


def _classify_roles(ingredient_name: str) -> List[str]:
    """Identify the functional roles of an ingredient."""
    norm = normalize(ingredient_name, strip_noise=True)
    roles = []
    for role, members in ROLE_MAP.items():
        for member in members:
            if member in norm or norm in member:
                roles.append(role)
                break
    return roles if roles else ["flavor"]  # Default role


@dataclass
class SubstitutionCandidate:
    """A single substitution option with evidence."""
    substitute_name: str
    roles_fulfilled: List[str]
    supporting_recipe_ids: List[str]
    confidence: str  # "high", "medium", "low"
    evidence_type: str  # "direct", "co-occurrence", "role-based", "external"
    notes: str = ""


@dataclass
class SubstitutionResult:
    """Complete substitution result for a recipe modification."""
    original_recipe: Recipe
    ingredient_replaced: str
    ingredient_roles: List[str]
    candidates: List[SubstitutionCandidate]
    modified_ingredients: List[str]
    modified_instructions: str
    impact_notes: str
    evidence_summary: str
    restriction: Optional[str] = None


class SubstitutionEngine:
    """Evidence-based ingredient substitution using the local recipe dataset."""

    def __init__(self, recipes: List[Recipe], inverted_index: Optional[Dict[str, List[str]]] = None):
        self._recipes = recipes
        self._by_id: Dict[str, Recipe] = {r.recipe_id: r for r in recipes}

        if inverted_index is not None:
            self._ing_to_recipes = inverted_index
        else:
            # Build ingredient → recipe_ids mapping
            self._ing_to_recipes = {}
            for r in recipes:
                for ing in r.ingredient_names:
                    self._ing_to_recipes.setdefault(ing, []).append(r.recipe_id)

    def _find_similar_recipes(
        self,
        recipe: Recipe,
        min_overlap: int = 3,
    ) -> List[Tuple[Recipe, int]]:
        """Find recipes sharing at least `min_overlap` ingredients (optimized via inverted index)."""
        target_ings = set(recipe.ingredient_names)
        # Fast candidate gathering: only recipes sharing at least one ingredient
        candidate_counts: Dict[str, int] = {}
        for ing in target_ings:
            for rid in self._ing_to_recipes.get(ing, []):
                if rid != recipe.recipe_id:
                    candidate_counts[rid] = candidate_counts.get(rid, 0) + 1

        similar = []
        for rid, count in candidate_counts.items():
            if count >= min_overlap and rid in self._by_id:
                similar.append((self._by_id[rid], count))

        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:100]  # Cap for performance

    def _find_direct_evidence(
        self,
        recipe: Recipe,
        ingredient_to_replace: str,
        restriction: Optional[str] = None,
    ) -> List[SubstitutionCandidate]:
        """
        Strategy 1: Find similar recipes that DON'T use the target ingredient.
        Whatever they use instead is direct evidence.
        """
        norm_target = normalize(ingredient_to_replace, strip_noise=True)
        target_roles = _classify_roles(ingredient_to_replace)
        target_ings = set(recipe.ingredient_names) - {norm_target}

        # Restriction expansion
        restricted_set: Set[str] = set()
        if restriction:
            from engine.matching_engine import INGREDIENT_GROUPS, _expand_exclusions
            restricted_set = _expand_exclusions([restriction])
        restricted_set.add(norm_target)

        similar = self._find_similar_recipes(recipe, min_overlap=2)
        substitute_evidence: Dict[str, List[str]] = {}

        for sim_recipe, overlap in similar:
            sim_ings = set(sim_recipe.ingredient_names)

            # Skip if similar recipe also uses the ingredient to replace
            if any(norm_target in si or si in norm_target for si in sim_ings):
                continue

            # Skip if similar recipe uses a restricted ingredient
            if restricted_set and any(
                any(rs in si or si in rs for rs in restricted_set) for si in sim_ings
            ):
                continue

            # Find ingredients in similar recipe not in our recipe → potential substitutes
            novel_ings = sim_ings - target_ings
            for ni in novel_ings:
                ni_roles = _classify_roles(ni)
                # Only consider if it fulfills at least one role of the replaced ingredient
                if set(target_roles) & set(ni_roles):
                    substitute_evidence.setdefault(ni, []).append(sim_recipe.recipe_id)

        candidates = []
        for sub_name, recipe_ids in sorted(substitute_evidence.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(recipe_ids)
            if count >= 3:
                conf = "high"
            elif count >= 1:
                conf = "medium"
            else:
                conf = "low"

            candidates.append(SubstitutionCandidate(
                substitute_name=sub_name,
                roles_fulfilled=_classify_roles(sub_name),
                supporting_recipe_ids=recipe_ids[:5],
                confidence=conf,
                evidence_type="direct",
                notes=f"Found in {count} similar recipe(s) that omit {ingredient_to_replace}.",
            ))

        return candidates[:5]

    def _find_role_based_evidence(
        self,
        ingredient_to_replace: str,
        restriction: Optional[str] = None,
    ) -> List[SubstitutionCandidate]:
        """
        Strategy 3: Find ingredients that serve the same functional role.
        """
        target_roles = _classify_roles(ingredient_to_replace)
        norm_target = normalize(ingredient_to_replace, strip_noise=True)

        restricted_set: Set[str] = {norm_target}
        if restriction:
            from engine.matching_engine import _expand_exclusions
            restricted_set.update(_expand_exclusions([restriction]))

        candidates = []
        for role in target_roles:
            role_members = ROLE_MAP.get(role, [])
            for member in role_members:
                norm_member = normalize(member, strip_noise=True)
                if norm_member == norm_target:
                    continue
                if any(rs in norm_member or norm_member in rs for rs in restricted_set):
                    continue

                # Count how many recipes use this ingredient
                usage_count = len(self._ing_to_recipes.get(norm_member, []))
                if usage_count == 0:
                    continue

                recipe_ids = self._ing_to_recipes.get(norm_member, [])[:5]
                candidates.append(SubstitutionCandidate(
                    substitute_name=member,
                    roles_fulfilled=[role],
                    supporting_recipe_ids=recipe_ids,
                    confidence="low",
                    evidence_type="role-based",
                    notes=f"Serves the same '{role}' role. Used in {usage_count} dataset recipe(s). "
                          f"Evidence is indirect — no direct comparison with original recipe.",
                ))

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c.substitute_name not in seen:
                seen.add(c.substitute_name)
                unique.append(c)
        return unique[:5]

    def find_substitution(
        self,
        recipe: Recipe,
        ingredient_to_replace: str,
        restriction: Optional[str] = None,
    ) -> SubstitutionResult:
        """
        Find evidence-based substitutions for an ingredient in a recipe.

        Parameters
        ----------
        recipe : Recipe
            The target recipe.
        ingredient_to_replace : str
            The ingredient to replace.
        restriction : str, optional
            Dietary restriction (e.g., "dairy", "eggs", "nuts").
        """
        roles = _classify_roles(ingredient_to_replace)
        norm_target = normalize(ingredient_to_replace, strip_noise=True)

        # Gather evidence from multiple strategies
        direct = self._find_direct_evidence(recipe, ingredient_to_replace, restriction)
        role_based = self._find_role_based_evidence(ingredient_to_replace, restriction)

        # Merge candidates (direct evidence first)
        seen = set()
        all_candidates = []
        for c in direct + role_based:
            if c.substitute_name not in seen:
                seen.add(c.substitute_name)
                all_candidates.append(c)

        # Build modified ingredient list
        best_sub = all_candidates[0].substitute_name if all_candidates else "[no substitute found]"
        modified_ingredients = []
        for raw_ing in recipe.ingredients_raw:
            if norm_target in normalize(raw_ing, strip_noise=True):
                modified_ingredients.append(f"{best_sub} (substituted for {ingredient_to_replace})")
            else:
                modified_ingredients.append(raw_ing)

        # Build modified instructions
        modified_instructions = recipe.instructions
        if all_candidates:
            modified_instructions = recipe.instructions.replace(
                ingredient_to_replace, best_sub
            )
            # Also try with normalized name
            for raw in recipe.ingredients_raw:
                if norm_target in raw.lower():
                    # Extract the original name from raw ingredient
                    modified_instructions = modified_instructions.replace(
                        ingredient_to_replace.lower(), best_sub
                    )

        # Impact assessment
        impact_parts = []
        for role in roles:
            if role == "binding":
                impact_parts.append("May affect structure/binding. Monitor for texture changes.")
            elif role == "moisture":
                impact_parts.append("May alter moisture level. Adjust liquid quantities if needed.")
            elif role == "fat":
                impact_parts.append("May change richness and mouthfeel.")
            elif role == "leavening":
                impact_parts.append("May affect rise. Consider adjusting leavening agents.")
            elif role == "sweetener":
                impact_parts.append("May change sweetness level and caramelization.")
            elif role == "flavor":
                impact_parts.append("Will change the flavor profile.")
            elif role == "thickener":
                impact_parts.append("May affect consistency. Adjust ratios as needed.")

        impact = " ".join(impact_parts) if impact_parts else "Impact uncertain — test with small batch."
        impact += " NOTE: These are predictions based on ingredient roles, not verified outcomes."

        # Evidence summary
        direct_count = len([c for c in all_candidates if c.evidence_type == "direct"])
        role_count = len([c for c in all_candidates if c.evidence_type == "role-based"])
        evidence_parts = []
        if direct_count:
            evidence_parts.append(f"{direct_count} substitute(s) from direct recipe evidence")
        if role_count:
            evidence_parts.append(f"{role_count} substitute(s) from role-based analysis")
        if not evidence_parts:
            evidence_parts.append("No reliable substitution evidence found in dataset")
        evidence_summary = ". ".join(evidence_parts) + "."

        return SubstitutionResult(
            original_recipe=recipe,
            ingredient_replaced=ingredient_to_replace,
            ingredient_roles=roles,
            candidates=all_candidates,
            modified_ingredients=modified_ingredients,
            modified_instructions=modified_instructions,
            impact_notes=impact,
            evidence_summary=evidence_summary,
            restriction=restriction,
        )

    def get_recipe_by_id(self, recipe_id: str) -> Optional[Recipe]:
        return self._by_id.get(recipe_id)
