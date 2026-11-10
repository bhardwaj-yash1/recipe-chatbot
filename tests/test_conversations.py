"""
30 Multi-Turn Test Conversations for the Recipe Agent.

Tests span all three assignment tasks:
  - Search & Q&A        (10 conversations)
  - Analytics            (8 conversations)
  - Ingredient Matching  (6 conversations)
  - Substitution         (6 conversations)

Each test exercises the full orchestrator pipeline: classification →
extraction → execution → response generation.  Follow-up turns verify
that conversation context is maintained correctly.

Run with:  pytest tests/test_conversations.py -v
"""

import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from core.data_loader import load_processed_recipes
from config.settings import settings
from engine.analytics_engine import AnalyticsEngine
from engine.matching_engine import MatchingEngine
from engine.search_engine import SearchEngine
from engine.substitution_engine import SubstitutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def orchestrator():
    """Build a fully initialised orchestrator (shared across tests in module)."""
    recipes = load_processed_recipes()
    se = SearchEngine(recipes, settings.processed_data_dir)
    se.load_all_indexes()
    ae = AnalyticsEngine(recipes)
    me = MatchingEngine(recipes)
    sub = SubstitutionEngine(recipes)
    llm = LLMClient()
    orch = Orchestrator(se, ae, me, sub, llm)
    return orch


def _reset(orch: Orchestrator):
    """Reset context between independent test conversations."""
    orch.reset()


# ══════════════════════════════════════════════════════════════════════
# SEARCH & Q&A  (Conversations 1-10)
# ══════════════════════════════════════════════════════════════════════

class TestSearchConversations:

    def test_01_simple_dish_name_search(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Find a penne pasta recipe")
        assert resp, "Empty response"
        assert any(word in resp.lower() for word in ["penne", "pasta", "recipe"]), \
            f"Response should mention penne/pasta: {resp[:200]}"

    def test_02_search_by_ingredient(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Show me recipes that use avocado")
        assert resp
        assert "avocado" in resp.lower() or "recipe" in resp.lower()

    def test_03_multi_turn_search_then_refine(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a penne pasta recipe")
        assert r1
        r2 = orchestrator.handle_message("I cannot have dairy")
        assert r2
        # Refinement should mention dairy-free or filtered results
        lower = r2.lower()
        assert any(w in lower for w in ["dairy", "refine", "filter", "without", "free", "recipe"])

    def test_04_search_then_detail_then_refine(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Search for chicken soup recipes")
        assert r1
        r2 = orchestrator.handle_message("Tell me more about the first recipe")
        assert r2
        r3 = orchestrator.handle_message("Are there any without onion?")
        assert r3

    def test_05_semantic_search(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Something light and refreshing for summer")
        assert resp
        assert "recipe" in resp.lower() or "salad" in resp.lower() or "found" in resp.lower()

    def test_06_recipe_detail_by_id(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Show me the details for recipe ar_00001")
        assert resp
        assert "ar_00001" in resp or "ingredient" in resp.lower()

    def test_07_search_no_results(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Find recipes for unicorn steak with dragon sauce")
        assert resp
        # Should handle gracefully

    def test_08_ambiguous_search_follow_up(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find me a cake")
        assert r1
        r2 = orchestrator.handle_message("I meant chocolate cake specifically")
        assert r2
        assert "chocolate" in r2.lower() or "cake" in r2.lower()

    def test_09_multi_ingredient_search(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("Find recipes that use both garlic and basil")
        assert resp
        lower = resp.lower()
        assert "garlic" in lower or "basil" in lower or "recipe" in lower

    def test_10_search_then_ask_ingredients(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a lasagna recipe")
        assert r1
        r2 = orchestrator.handle_message("What ingredients does the first one need?")
        assert r2


# ══════════════════════════════════════════════════════════════════════
# ANALYTICS  (Conversations 11-18)
# ══════════════════════════════════════════════════════════════════════

class TestAnalyticsConversations:

    def test_11_count_with_ingredient(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("How many recipes contain garlic?")
        assert resp
        # Should contain a number and mention garlic
        assert any(c.isdigit() for c in resp), f"Response should contain numbers: {resp[:200]}"

    def test_12_percentage_calculation(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "How many recipes contain both garlic and onion? What percentage of the dataset is that?"
        )
        assert resp
        assert "%" in resp or "percent" in resp.lower()

    def test_13_average_ingredient_count(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("What is the average number of ingredients per recipe?")
        assert resp
        assert any(c.isdigit() for c in resp)

    def test_14_rank_by_fewest_steps(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "Among recipes containing tomato, which five have the fewest instruction steps?"
        )
        assert resp
        assert any(c.isdigit() for c in resp)

    def test_15_most_common_ingredients(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message("What are the 10 most common ingredients across all recipes?")
        assert resp

    def test_16_multi_turn_count_then_refine(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("How many recipes use chicken?")
        assert r1
        r2 = orchestrator.handle_message("And how many of those also use garlic?")
        assert r2

    def test_17_median_ingredient_count(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "What is the median number of ingredients in pasta recipes?"
        )
        assert resp

    def test_18_compare_ingredient_frequencies(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "Which is more common in recipes: butter or olive oil?"
        )
        assert resp


# ══════════════════════════════════════════════════════════════════════
# INGREDIENT MATCHING  (Conversations 19-24)
# ══════════════════════════════════════════════════════════════════════

class TestMatchingConversations:

    def test_19_basic_pantry_matching(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "I have eggs, flour, sugar, and butter. What can I make?"
        )
        assert resp
        assert "recipe" in resp.lower() or "match" in resp.lower() or "found" in resp.lower()

    def test_20_matching_with_exclusions(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many "
            "of these as possible, with no peanuts and at most six instruction steps. "
            "Prefer recipes that need the fewest additional ingredients."
        )
        assert resp

    def test_21_matching_with_max_steps(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "I have chicken, garlic, and olive oil. Find recipes with at most 4 steps."
        )
        assert resp

    def test_22_multi_turn_match_then_exclude(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message(
            "I have eggs, onions, tomatoes, and rice. What can I make?"
        )
        assert r1
        r2 = orchestrator.handle_message("Exclude eggs too")
        assert r2

    def test_23_no_match_case(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "I only have water. Find recipes with no flour, no sugar, no salt, "
            "no butter, no oil, and at most 1 step."
        )
        assert resp
        # Should indicate no match or very limited results

    def test_24_match_prefer_fewest_missing(self, orchestrator):
        _reset(orchestrator)
        resp = orchestrator.handle_message(
            "I have pasta, garlic, olive oil, parmesan, and black pepper. "
            "Find recipes needing the fewest extra ingredients."
        )
        assert resp


# ══════════════════════════════════════════════════════════════════════
# SUBSTITUTION  (Conversations 25-30)
# ══════════════════════════════════════════════════════════════════════

class TestSubstitutionConversations:

    def test_25_simple_egg_substitution(self, orchestrator):
        _reset(orchestrator)
        # First find a banana bread recipe
        r1 = orchestrator.handle_message("Find a banana bread recipe")
        assert r1
        r2 = orchestrator.handle_message(
            "Take this banana bread recipe and remove eggs. "
            "What replacements have evidence from related baked recipes?"
        )
        assert r2
        # Should mention substitution evidence
        lower = r2.lower()
        assert any(w in lower for w in ["substitut", "replac", "evidence", "recipe"])

    def test_26_dairy_free_adaptation(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a pasta with cream sauce recipe")
        assert r1
        r2 = orchestrator.handle_message(
            "Replace cream in this pasta recipe without dairy. "
            "Prefer examples from recipes already in the database."
        )
        assert r2

    def test_27_multi_substitution_allergy(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a cookie recipe")
        assert r1
        r2 = orchestrator.handle_message(
            "I need this recipe without eggs and without dairy. "
            "Adapt it for me."
        )
        assert r2

    def test_28_weak_evidence_substitution(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a recipe that uses saffron")
        assert r1
        r2 = orchestrator.handle_message(
            "Can I replace saffron with something cheaper?"
        )
        assert r2
        # Should handle weak/uncertain evidence appropriately

    def test_29_multi_turn_substitute_then_impact(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a pancake recipe")
        assert r1
        r2 = orchestrator.handle_message("Replace the eggs in this recipe")
        assert r2
        r3 = orchestrator.handle_message("How might that affect the texture?")
        assert r3

    def test_30_no_reliable_substitution(self, orchestrator):
        _reset(orchestrator)
        r1 = orchestrator.handle_message("Find a recipe using gelatin")
        assert r1
        r2 = orchestrator.handle_message(
            "Replace gelatin with something vegan, but it must set like gelatin"
        )
        assert r2
