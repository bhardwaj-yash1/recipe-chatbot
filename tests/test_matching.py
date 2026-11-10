"""
Unit tests for the Matching Engine — verifies ingredient-based matching,
constraint enforcement (hard exclusions and max steps), and ranking logic.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_processed_recipes
from engine.matching_engine import MatchingEngine, INGREDIENT_GROUPS


@pytest.fixture(scope="module")
def matching_engine():
    recipes = load_processed_recipes()
    return MatchingEngine(recipes)


class TestMatchingBasics:

    def test_match_returns_results(self, matching_engine):
        results = matching_engine.match_recipes(
            available=["chicken", "garlic", "onion", "olive oil"],
            limit=5,
        )
        assert len(results) > 0
        assert len(results) <= 5
        for m in results:
            assert m.recipe is not None
            assert m.recipe.recipe_id
            assert m.score is not None
            assert m.rank_explanation

    def test_explainable_scoring(self, matching_engine):
        results = matching_engine.match_recipes(
            available=["pasta", "tomato", "basil", "garlic", "olive oil"],
            limit=3,
        )
        for m in results:
            assert "Coverage:" in m.rank_explanation
            assert isinstance(m.matched_ingredients, list)
            assert isinstance(m.missing_ingredients, list)


class TestHardConstraintEnforcement:

    def test_strict_exclusion_peanut(self, matching_engine):
        results = matching_engine.match_recipes(
            available=["chicken", "rice", "soy sauce", "ginger"],
            excluded=["peanut", "peanuts"],
            limit=10,
        )
        for m in results:
            ing_names_lower = [i.lower() for i in m.recipe.ingredient_names]
            assert not any("peanut" in i for i in ing_names_lower), (
                f"Recipe {m.recipe.recipe_id} violated exclusion constraint!"
            )

    def test_strict_exclusion_dairy_group(self, matching_engine):
        results = matching_engine.match_recipes(
            available=["chicken", "potato", "onion", "garlic", "olive oil"],
            excluded=["dairy"],
            limit=10,
        )
        dairy_items = INGREDIENT_GROUPS["dairy"]
        for m in results:
            for ing in m.recipe.ingredient_names:
                assert ing not in dairy_items, (
                    f"Recipe {m.recipe.recipe_id} contains dairy item {ing}!"
                )

    def test_max_steps_constraint(self, matching_engine):
        max_s = 4
        results = matching_engine.match_recipes(
            available=["egg", "bread", "butter"],
            max_steps=max_s,
            limit=10,
        )
        for m in results:
            assert len(m.recipe.instruction_steps) <= max_s, (
                f"Recipe {m.recipe.recipe_id} has {len(m.recipe.instruction_steps)} steps, exceeds {max_s}!"
            )


class TestEdgeCasesAndNoMatch:

    def test_impossible_constraints_return_empty(self, matching_engine):
        # Impossible: only water, exclude everything
        results = matching_engine.match_recipes(
            available=["water"],
            excluded=["dairy", "gluten", "sugar", "salt", "oil", "meat", "egg", "onion", "garlic"],
            max_steps=1,
            limit=5,
        )
        assert isinstance(results, list)

    def test_closest_with_violations(self, matching_engine):
        # When no matches or few matches, closest_with_violations returns explicitly marked items
        closest = matching_engine.closest_with_violations(
            available=["chicken", "rice"],
            excluded=["onion", "garlic", "salt"],
            max_steps=2,
            limit=3,
        )
        for m in closest:
            assert m.violates_constraint is True
            assert len(m.violation_details) > 0
