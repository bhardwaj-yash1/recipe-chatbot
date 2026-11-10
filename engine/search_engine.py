"""
Search Engine — hybrid search over the recipe corpus.

Combines FAISS dense semantic search, BM25 sparse keyword search,
and an inverted ingredient index.  All operations are fully deterministic;
no LLM calls are made here.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import settings
from core.models import Recipe
from core.normalizer import normalize

logger = logging.getLogger(__name__)


class SearchEngine:
    """Deterministic hybrid search over the recipe corpus."""

    def __init__(self, recipes: List[Recipe], processed_dir: Path | None = None):
        self._processed = processed_dir or settings.processed_data_dir
        self._recipes = recipes
        self._id_to_recipe: Dict[str, Recipe] = {r.recipe_id: r for r in recipes}

        # Lazy-loaded indexes
        self._faiss_index: Optional[faiss.Index] = None
        self._faiss_id_map: Optional[Dict[int, str]] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: Optional[List[str]] = None
        self._inverted: Optional[Dict[str, List[str]]] = None

    # ------------------------------------------------------------------
    # Index loading (lazy)
    # ------------------------------------------------------------------
    def _load_faiss(self):
        if self._faiss_index is not None:
            return
        idx_path = self._processed / "faiss_index.bin"
        map_path = self._processed / "faiss_id_map.json"
        if not idx_path.exists() or not map_path.exists():
            logger.warning("FAISS index not found at %s. Semantic search will fall back to BM25.", idx_path)
            return
        self._faiss_index = faiss.read_index(str(idx_path))
        with open(map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._faiss_id_map = {int(k): v for k, v in raw.items()}
        self._embedder = SentenceTransformer(settings.embedding_model)
        logger.info("FAISS index loaded (%d vectors)", self._faiss_index.ntotal)

    def _load_bm25(self):
        if self._bm25 is not None:
            return
        with open(self._processed / "bm25_index.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        with open(self._processed / "bm25_recipe_ids.json", "r", encoding="utf-8") as f:
            self._bm25_ids = json.load(f)
        logger.info("BM25 index loaded (%d docs)", len(self._bm25_ids))

    def _load_inverted(self):
        if self._inverted is not None:
            return
        with open(self._processed / "inverted_ingredient_index.json", "r", encoding="utf-8") as f:
            self._inverted = json.load(f)
        logger.info("Inverted index loaded (%d ingredients)", len(self._inverted))

    def load_all_indexes(self):
        """Pre-load every index (call at startup to avoid first-query latency)."""
        self._load_faiss()
        self._load_bm25()
        self._load_inverted()

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------
    def search_by_name(self, query: str, limit: int = 5) -> List[Recipe]:
        """BM25 keyword search over recipe titles with quality filtering and overlap boost."""
        self._load_bm25()
        tokens = [t.lower() for t in query.split() if len(t) > 1]
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:200]

        q_lower = query.lower()
        candidates = []
        for i in top_idx:
            if scores[i] <= 0:
                break
            rid = self._bm25_ids[i]
            r = self._id_to_recipe.get(rid)
            if not r or len(r.ingredients_raw) < 2 or len(r.instruction_steps) < 1 or len(r.instructions.strip()) <= 20:
                continue

            title_lower = r.title.lower()
            title_words = set(title_lower.split())
            overlap = len(set(tokens) & title_words)
            boost = 1.0 + (overlap * 2.0)
            if q_lower in title_lower:
                boost += 5.0
            # Sauce clash prevention (e.g. Alfredo when asking for red/tomato)
            if ("red" in q_lower or "tomato" in q_lower) and "alfredo" in title_lower:
                boost *= 0.1
            if "white" in q_lower and ("red" in title_lower or "tomato" in title_lower):
                boost *= 0.1

            candidates.append((r, scores[i] * boost))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in candidates[:limit]]

    def search_by_ingredients(
        self,
        ingredients: List[str],
        mode: str = "all",
        limit: int = 10,
    ) -> List[Recipe]:
        """
        Search using the inverted ingredient index.

        Parameters
        ----------
        ingredients : list[str]
            Ingredient names (will be normalised).
        mode : 'all' | 'any'
            'all' = intersection (recipes must contain ALL ingredients).
            'any' = union (recipes containing ANY ingredient).
        limit : int
            Maximum results to return.
        """
        self._load_inverted()
        norm_ings = [normalize(ing, strip_noise=True) for ing in ingredients]

        sets: List[Set[str]] = []
        for ni in norm_ings:
            ids = set(self._inverted.get(ni, []))
            # Also try partial matching
            if not ids:
                for key, val in self._inverted.items():
                    if ni in key or key in ni:
                        ids.update(val)
            sets.append(ids)

        if not sets:
            return []

        if mode == "all":
            matched_ids = sets[0]
            for s in sets[1:]:
                matched_ids = matched_ids & s
        else:
            matched_ids = set()
            for s in sets:
                matched_ids = matched_ids | s

        # Rank by how many of the query ingredients the recipe contains
        results = []
        for rid in matched_ids:
            if rid in self._id_to_recipe:
                results.append(self._id_to_recipe[rid])

        # Sort by number of matched query ingredients descending
        def _match_count(r: Recipe) -> int:
            r_ings = set(r.ingredient_names)
            return sum(1 for ni in norm_ings if ni in r_ings)

        results.sort(key=_match_count, reverse=True)
        return results[:limit]

    def search_semantic(self, query: str, limit: int = 5) -> List[Recipe]:
        """FAISS dense semantic search with quality filtering."""
        self._load_faiss()
        if self._faiss_index is None:
            return self.search_by_name(query, limit=limit)
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        fetch_k = min(len(self._recipes), limit * 5)
        scores, indices = self._faiss_index.search(q_emb, fetch_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rid = self._faiss_id_map.get(idx)
            if rid and rid in self._id_to_recipe:
                r = self._id_to_recipe[rid]
                if len(r.ingredients_raw) >= 2 and len(r.instruction_steps) >= 1 and len(r.instructions.strip()) > 20:
                    results.append(r)
                    if len(results) >= limit:
                        break
        return results

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        bm25_weight: float = 0.4,
        faiss_weight: float = 0.6,
    ) -> List[Recipe]:
        """
        Combined BM25 + FAISS search with configurable weights.

        Scores are min-max normalised independently before blending.
        """
        self._load_bm25()
        self._load_faiss()

        if self._faiss_index is None:
            return self.search_by_name(query, limit=limit)

        # BM25 scores
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)

        # FAISS scores
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        fetch_k = min(len(self._recipes), 500)
        faiss_scores_raw, faiss_indices = self._faiss_index.search(q_emb, fetch_k)
        faiss_score_map: Dict[str, float] = {}
        for score, idx in zip(faiss_scores_raw[0], faiss_indices[0]):
            if idx >= 0:
                rid = self._faiss_id_map.get(idx, "")
                faiss_score_map[rid] = float(score)

        # Normalise BM25
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_min = bm25_scores.min()
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

        # Normalise FAISS
        faiss_vals = list(faiss_score_map.values()) if faiss_score_map else [0]
        faiss_max = max(faiss_vals) if max(faiss_vals) > 0 else 1.0
        faiss_min = min(faiss_vals)
        faiss_range = faiss_max - faiss_min if faiss_max != faiss_min else 1.0

        # Blend
        combined: Dict[str, float] = {}
        for i, rid in enumerate(self._bm25_ids):
            bm25_norm = (bm25_scores[i] - bm25_min) / bm25_range
            faiss_norm = (faiss_score_map.get(rid, faiss_min) - faiss_min) / faiss_range
            combined[rid] = bm25_weight * bm25_norm + faiss_weight * faiss_norm

        top_ids = sorted(combined, key=combined.get, reverse=True)
        results = []
        for rid in top_ids:
            if rid in self._id_to_recipe:
                r = self._id_to_recipe[rid]
                if len(r.ingredients_raw) >= 2 and len(r.instruction_steps) >= 1 and len(r.instructions.strip()) > 20:
                    results.append(r)
                    if len(results) >= limit:
                        break
        return results

    def refine_results(
        self,
        recipe_ids: List[str],
        exclude_ingredients: Optional[List[str]] = None,
        include_ingredients: Optional[List[str]] = None,
    ) -> List[Recipe]:
        """
        Filter an existing result set by ingredient constraints.

        Parameters
        ----------
        recipe_ids : list[str]
            IDs to filter.
        exclude_ingredients : list[str], optional
            Remove recipes containing ANY of these.
        include_ingredients : list[str], optional
            Keep only recipes containing ALL of these.
        """
        recipes = [self._id_to_recipe[rid] for rid in recipe_ids if rid in self._id_to_recipe]

        if exclude_ingredients:
            norm_excl = {normalize(e, strip_noise=True) for e in exclude_ingredients}
            recipes = [
                r for r in recipes
                if not any(ni in norm_excl or any(ex in ni for ex in norm_excl) for ni in r.ingredient_names)
            ]

        if include_ingredients:
            norm_incl = {normalize(i, strip_noise=True) for i in include_ingredients}
            recipes = [
                r for r in recipes
                if all(any(ni_item in rin or rin in ni_item for rin in r.ingredient_names) for ni_item in norm_incl)
            ]

        return recipes

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Retrieve a single recipe by its stable ID."""
        return self._id_to_recipe.get(recipe_id)

    def get_recipes_by_ids(self, ids: List[str]) -> List[Recipe]:
        """Retrieve multiple recipes by their stable IDs."""
        return [self._id_to_recipe[rid] for rid in ids if rid in self._id_to_recipe]

    def get_all_recipes(self) -> List[Recipe]:
        """Return all loaded recipes."""
        return self._recipes

    @property
    def total_recipes(self) -> int:
        return len(self._recipes)
