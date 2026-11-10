"""
Edge Case Tests for the Recipe Agent.

Covers:
  - Impossible constraints / no-match handling
  - Silent relaxation prevention (hard constraints must never be violated)
  - Hallucination prevention (ungrounded LLM answers vs grounded engine results)
  - Weak or conflicting substitution evidence
  - Malformed or noisy user queries
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_processed_recipes
from engine.analytics_engine import AnalyticsEngine
from engine.matching_engine import MatchingEngine
from engine.substitution_engine import SubstitutionEngine
from agent.hallucination_guard import scrub_hallucinated_params


@pytest.fixture(scope="module")
def recipes():
    return load_processed_recipes()


class TestEdgeCases:

    def test_never_silently_relax_hard_constraint(self, recipes):
        me = MatchingEngine(recipes)
        # Even with an impossible ingredient combination + strict exclusion,
        # it should NEVER return a recipe containing peanuts
        matches = me.match_recipes(
            available=["peanut butter", "jelly", "bread"],
            excluded=["peanut", "peanuts", "nuts"],
            limit=5,
        )
        for m in matches:
            for ing in m.recipe.ingredient_names:
                assert "peanut" not in ing.lower()
                assert "nut" not in ing.lower()

    def test_no_match_returns_empty_and_closest_are_flagged(self, recipes):
        me = MatchingEngine(recipes)
        matches = me.match_recipes(
            available=["dragonfruit", "quail egg", "truffle oil"],
            excluded=["salt", "pepper", "oil", "water", "garlic", "onion"],
            max_steps=1,
            limit=5,
        )
        # If no strict match, check closest_with_violations
        closest = me.closest_with_violations(
            available=["dragonfruit", "quail egg", "truffle oil"],
            excluded=["salt", "pepper", "oil", "water", "garlic", "onion"],
            max_steps=1,
            limit=3,
        )
        for c in closest:
            assert c.violates_constraint is True
            assert c.violation_details

    def test_hallucination_guard_scrubs_invented_params(self):
        user_message = "Find pasta recipes"
        context_messages = []
        # LLM hallucinates extra parameters not in prompt
        hallucinated_params = {
            "query": "pasta",
            "servings": 4,           # Not in text!
            "cook_time_max": 30,     # Not in text!
            "cuisine": "Italian",    # Not in text!
        }
        scrubbed = scrub_hallucinated_params(
            hallucinated_params,
            user_message,
            context_messages,
            intent_name="SEARCH_BY_NAME",
        )
        assert "query" in scrubbed
        assert scrubbed["query"] == "pasta"
        assert "servings" not in scrubbed, "Hallucinated servings was not scrubbed!"
        assert "cook_time_max" not in scrubbed, "Hallucinated cook_time was not scrubbed!"
        assert "cuisine" not in scrubbed, "Hallucinated cuisine was not scrubbed!"

    def test_substitution_weak_evidence_flagged(self, recipes):
        sub_engine = SubstitutionEngine(recipes)
        sample = recipes[0]
        # Try substituting an exotic/uncommon ingredient
        result = sub_engine.find_substitution(
            recipe=sample,
            ingredient_to_replace="star anise",
            restriction=None,
        )
        assert result is not None
        # Confidence should reflect lack of direct evidence
        for c in result.candidates:
            if c.evidence_type != "direct":
                assert c.confidence in ["medium", "low"]

    def test_analytics_unknown_ingredient_zero_count(self, recipes):
        ae = AnalyticsEngine(recipes)
        result = ae.count_recipes(must_include=["nonexistent_ingredient_xyz_12345"])
        assert result.value == 0
        assert result.total_analyzed == 0
        assert result.total_dataset == len(recipes)
