"""
Analytics Engine — deterministic statistics over the recipe corpus.

All calculations are performed with Pandas on the full filtered dataset.
No LLM calls.  Every result includes the filters applied, denominator,
and calculation method for full transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from core.models import Recipe
from core.normalizer import normalize

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsResult:
    """Container for every analytics response — enforces transparency."""
    metric: str
    value: Any
    filters_applied: str
    total_analyzed: int
    total_dataset: int
    method: str
    details: Optional[Dict] = None


def _ingredient_matches(query_ing: str, recipe_ing: str) -> bool:
    """Robust matching between query ingredient and recipe ingredient."""
    if not query_ing or not recipe_ing:
        return False
    q = query_ing.lower().strip()
    r = recipe_ing.lower().strip()
    if q == r or q in r:
        return True
    return r in q.split()


class AnalyticsEngine:
    """Pandas-powered deterministic analytics over the recipe corpus."""

    def __init__(self, recipes: List[Recipe]):
        self._recipes = recipes
        self._df = self._build_dataframe(recipes)
        self._total = len(recipes)

    @staticmethod
    def _build_dataframe(recipes: List[Recipe]) -> pd.DataFrame:
        rows = []
        for r in recipes:
            rows.append({
                "recipe_id": r.recipe_id,
                "source": r.source,
                "title": r.title,
                "ingredient_names": [n for n in r.ingredient_names if n and len(n.strip()) >= 2],
                "ingredient_count": len(r.ingredient_names),
                "step_count": len(r.instruction_steps),
                "ingredients_raw": r.ingredients_raw,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------
    def _filter_by_ingredients(
        self,
        df: pd.DataFrame,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Filter df rows by ingredient containment."""
        desc_parts = []

        if must_include:
            norm_inc = [normalize(i, strip_noise=True) for i in must_include if i]
            for ni in norm_inc:
                if not ni:
                    continue
                mask = df["ingredient_names"].apply(
                    lambda names: any(_ingredient_matches(ni, n) for n in names)
                )
                df = df[mask]
            desc_parts.append(f"include=[{', '.join(must_include)}]")

        if must_exclude:
            norm_exc = [normalize(e, strip_noise=True) for e in must_exclude if e]
            for ne in norm_exc:
                if not ne:
                    continue
                mask = df["ingredient_names"].apply(
                    lambda names: not any(_ingredient_matches(ne, n) for n in names)
                )
                df = df[mask]
            desc_parts.append(f"exclude=[{', '.join(must_exclude)}]")

        return df, "; ".join(desc_parts) if desc_parts else "none"

    def _filter_by_title(self, df: pd.DataFrame, keyword: str) -> Tuple[pd.DataFrame, str]:
        mask = df["title"].str.contains(keyword, case=False, na=False)
        return df[mask], f"title contains '{keyword}'"

    # ------------------------------------------------------------------
    # Public analytics methods
    # ------------------------------------------------------------------
    def count_recipes(
        self,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
        title_keyword: Optional[str] = None,
    ) -> AnalyticsResult:
        """Count recipes matching the given filters."""
        df = self._df.copy()
        filter_desc = "none"

        if title_keyword:
            df, fd = self._filter_by_title(df, title_keyword)
            filter_desc = fd

        if must_include or must_exclude:
            df, fd = self._filter_by_ingredients(df, must_include, must_exclude)
            filter_desc = fd if filter_desc == "none" else f"{filter_desc}; {fd}"

        return AnalyticsResult(
            metric="recipe_count",
            value=len(df),
            filters_applied=filter_desc,
            total_analyzed=len(df),
            total_dataset=self._total,
            method="DataFrame row count after applying filters",
        )

    def percentage(
        self,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
        title_keyword: Optional[str] = None,
    ) -> AnalyticsResult:
        """Calculate percentage of dataset matching the given filters."""
        count_result = self.count_recipes(must_include, must_exclude, title_keyword)
        pct = (count_result.value / self._total * 100) if self._total else 0.0
        return AnalyticsResult(
            metric="percentage",
            value=round(pct, 2),
            filters_applied=count_result.filters_applied,
            total_analyzed=count_result.total_analyzed,
            total_dataset=self._total,
            method=f"count({count_result.value}) / total({self._total}) × 100",
            details={"count": count_result.value, "total": self._total},
        )

    def ingredient_frequency(
        self,
        top_n: int = 10,
        must_include: Optional[List[str]] = None,
        title_keyword: Optional[str] = None,
    ) -> AnalyticsResult:
        """Find the most/least common ingredients in the (filtered) dataset."""
        df = self._df.copy()
        filter_desc = "none"

        if title_keyword:
            df, fd = self._filter_by_title(df, title_keyword)
            filter_desc = fd
        if must_include:
            df, fd = self._filter_by_ingredients(df, must_include=must_include)
            filter_desc = fd if filter_desc == "none" else f"{filter_desc}; {fd}"

        # Explode ingredient_names
        exploded = df.explode("ingredient_names")
        freq = exploded["ingredient_names"].value_counts().head(top_n)
        freq_list = [(name, int(count)) for name, count in freq.items()]

        return AnalyticsResult(
            metric="ingredient_frequency",
            value=freq_list,
            filters_applied=filter_desc,
            total_analyzed=len(df),
            total_dataset=self._total,
            method=f"Explode ingredient_names column → value_counts() → top {top_n}",
        )

    def ingredient_stats(
        self,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
        title_keyword: Optional[str] = None,
    ) -> AnalyticsResult:
        """Compute mean, median, min, max of ingredient counts."""
        df = self._df.copy()
        filter_desc = "none"

        if title_keyword:
            df, fd = self._filter_by_title(df, title_keyword)
            filter_desc = fd
        if must_include or must_exclude:
            df, fd = self._filter_by_ingredients(df, must_include, must_exclude)
            filter_desc = fd if filter_desc == "none" else f"{filter_desc}; {fd}"

        col = df["ingredient_count"]
        stats = {
            "mean": round(float(col.mean()), 2) if len(col) else 0,
            "median": float(col.median()) if len(col) else 0,
            "min": int(col.min()) if len(col) else 0,
            "max": int(col.max()) if len(col) else 0,
            "std": round(float(col.std()), 2) if len(col) else 0,
        }

        return AnalyticsResult(
            metric="ingredient_stats",
            value=stats,
            filters_applied=filter_desc,
            total_analyzed=len(df),
            total_dataset=self._total,
            method="Pandas .mean() / .median() / .min() / .max() on ingredient_count column",
        )

    def rank_by_steps(
        self,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None,
        title_keyword: Optional[str] = None,
        ascending: bool = True,
        limit: int = 5,
    ) -> AnalyticsResult:
        """Rank recipes by instruction step count."""
        df = self._df.copy()
        filter_desc = "none"

        if title_keyword:
            df, fd = self._filter_by_title(df, title_keyword)
            filter_desc = fd
        if must_include or must_exclude:
            df, fd = self._filter_by_ingredients(df, must_include, must_exclude)
            filter_desc = fd if filter_desc == "none" else f"{filter_desc}; {fd}"

        # Filter out zero-step recipes for meaningful ranking
        df = df[df["step_count"] > 0]

        sorted_df = df.sort_values("step_count", ascending=ascending).head(limit)
        results = []
        for _, row in sorted_df.iterrows():
            results.append({
                "recipe_id": row["recipe_id"],
                "title": row["title"],
                "step_count": int(row["step_count"]),
                "ingredient_count": int(row["ingredient_count"]),
            })

        order_desc = "ascending (fewest first)" if ascending else "descending (most first)"

        return AnalyticsResult(
            metric="rank_by_steps",
            value=results,
            filters_applied=filter_desc,
            total_analyzed=len(df),
            total_dataset=self._total,
            method=f"Sort by step_count {order_desc}, return top {limit}. "
                   f"Step count = number of instruction steps (split by newline/numbered list). "
                   f"Recipes with 0 steps excluded.",
        )
