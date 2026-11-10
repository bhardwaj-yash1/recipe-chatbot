"""
Unit tests for the Substitution Engine — verifies dataset-grounded substitution,
evidence collection, role analysis, and recipe adaptation.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_processed_recipes
from engine.substitution_engine import SubstitutionEngine


@pytest.fixture(scope="module")
def sub_engine():
    recipes = load_processed_recipes()
    return SubstitutionEngine(recipes)


class TestSubstitutionBasics:

    def test_find_substitution_eggs(self, sub_engine):
        # Pick a recipe that has eggs
        sample = None
        for r in sub_engine._recipes[:500]:
            if any("egg" in ing for ing in r.ingredient_names):
                sample = r
                break
        assert sample is not None

        result = sub_engine.find_substitution(
            recipe=sample,
            ingredient_to_replace="egg",
            restriction="vegan",
        )

        assert result is not None
        assert result.original_recipe.recipe_id == sample.recipe_id
        assert result.ingredient_replaced == "egg"
        assert len(result.candidates) > 0
        assert len(result.modified_ingredients) == len(sample.ingredients_raw)
        assert len(result.modified_instructions) > 0
        assert result.evidence_summary

    def test_find_substitution_dairy(self, sub_engine):
        # Pick a recipe that has milk or butter
        sample = None
        for r in sub_engine._recipes[:500]:
            if any("butter" in ing for ing in r.ingredient_names):
                sample = r
                break
        assert sample is not None

        result = sub_engine.find_substitution(
            recipe=sample,
            ingredient_to_replace="butter",
            restriction="dairy",
        )

        assert result is not None
        assert len(result.candidates) > 0
        for c in result.candidates:
            assert c.substitute_name
            assert c.confidence in ["high", "medium", "low"]
            assert c.evidence_type in ["direct", "co-occurrence", "role-based", "external"]

    def test_supporting_recipe_ids_cited(self, sub_engine):
        sample = sub_engine._recipes[0]
        first_ing = sample.ingredient_names[0] if sample.ingredient_names else "onion"
        result = sub_engine.find_substitution(
            recipe=sample,
            ingredient_to_replace=first_ing,
        )
        assert result is not None
        for cand in result.candidates:
            if cand.evidence_type == "direct":
                assert len(cand.supporting_recipe_ids) > 0
                for rid in cand.supporting_recipe_ids:
                    assert rid.startswith(("ar_", "epi_", "fn_"))
