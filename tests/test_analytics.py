"""
Unit tests for the Analytics Engine — verifies correctness of
all computations (counts, percentages, rankings, averages).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_processed_recipes
from engine.analytics_engine import AnalyticsEngine


@pytest.fixture(scope="module")
def engine():
    recipes = load_processed_recipes()
    return AnalyticsEngine(recipes)


class TestCounting:
    def test_total_count_no_filter(self, engine):
        result = engine.count_recipes()
        assert result.value > 100_000, f"Expected 125K+ recipes, got {result.value}"
        assert result.total_dataset == result.value

    def test_count_with_ingredient(self, engine):
        result = engine.count_recipes(must_include=["garlic"])
        assert result.value > 0
        assert result.value < result.total_dataset
        assert "garlic" in result.filters_applied.lower()

    def test_count_with_two_ingredients(self, engine):
        r1 = engine.count_recipes(must_include=["garlic"])
        r2 = engine.count_recipes(must_include=["garlic", "onion"])
        assert r2.value <= r1.value, "AND filter should return fewer/equal results"

    def test_count_with_exclusion(self, engine):
        r_all = engine.count_recipes()
        r_excl = engine.count_recipes(must_exclude=["sugar"])
        assert r_excl.value < r_all.value

    def test_count_title_filter(self, engine):
        result = engine.count_recipes(title_keyword="chocolate")
        assert result.value > 0
        assert "chocolate" in result.filters_applied.lower()


class TestPercentage:
    def test_percentage_range(self, engine):
        result = engine.percentage(must_include=["garlic"])
        assert 0 < result.value < 100
        assert result.details["total"] == result.total_dataset

    def test_percentage_with_count(self, engine):
        pct = engine.percentage(must_include=["garlic"])
        cnt = engine.count_recipes(must_include=["garlic"])
        expected_pct = round(cnt.value / pct.total_dataset * 100, 2)
        assert abs(pct.value - expected_pct) < 0.01


class TestIngredientStats:
    def test_stats_structure(self, engine):
        result = engine.ingredient_stats()
        stats = result.value
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] >= 0
        assert stats["max"] >= stats["mean"]

    def test_stats_filtered(self, engine):
        all_stats = engine.ingredient_stats()
        filtered = engine.ingredient_stats(must_include=["chocolate"])
        # Both should return valid stats
        assert all_stats.value["mean"] > 0
        assert filtered.value["mean"] > 0


class TestRanking:
    def test_rank_ascending(self, engine):
        result = engine.rank_by_steps(ascending=True, limit=5)
        steps = [r["step_count"] for r in result.value]
        assert steps == sorted(steps), "Steps should be ascending"
        assert len(steps) <= 5

    def test_rank_descending(self, engine):
        result = engine.rank_by_steps(ascending=False, limit=5)
        steps = [r["step_count"] for r in result.value]
        assert steps == sorted(steps, reverse=True), "Steps should be descending"

    def test_rank_with_filter(self, engine):
        result = engine.rank_by_steps(must_include=["tomato"], ascending=True, limit=5)
        assert len(result.value) > 0
        assert result.total_analyzed < result.total_dataset


class TestFrequency:
    def test_frequency_top_10(self, engine):
        result = engine.ingredient_frequency(top_n=10)
        assert len(result.value) == 10
        # Counts should be descending
        counts = [c for _, c in result.value]
        assert counts == sorted(counts, reverse=True)

    def test_frequency_filtered(self, engine):
        result = engine.ingredient_frequency(top_n=5, must_include=["chocolate"])
        assert len(result.value) <= 5
        assert result.total_analyzed < result.total_dataset
