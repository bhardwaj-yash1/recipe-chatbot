"""
Index builder for the Recipe Agent.

Builds three search indices from processed recipes:
  1. **FAISS vector index** – dense semantic search over ``title + ingredient_names``
     using sentence-transformers/all-MiniLM-L6-v2 embeddings.
  2. **BM25 index** – sparse keyword search (rank_bm25).
  3. **Inverted ingredient index** – ``ingredient_name → set(recipe_ids)``.

All indices are persisted to ``data/processed/``.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import settings
from core.data_loader import load_processed_recipes
from core.models import Recipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text representation for search
# ---------------------------------------------------------------------------
def _recipe_search_text(recipe: Recipe) -> str:
    """Build searchable text from a recipe: title + normalised ingredient names."""
    parts = [recipe.title]
    parts.extend(recipe.ingredient_names)
    return " ".join(parts).strip()


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# 1.  FAISS vector index
# ---------------------------------------------------------------------------
def build_faiss_index(
    recipes: List[Recipe],
    output_dir: Path | None = None,
    batch_size: int | None = None,
) -> Path:
    """
    Build and save a FAISS flat-IP (inner product) index.

    Parameters
    ----------
    recipes : list[Recipe]
        Processed recipe objects.
    output_dir : Path
        Where to write the index.  Defaults to ``data/processed/``.
    batch_size : int
        Encoding batch size.

    Returns
    -------
    Path
        Path to the saved FAISS index file.
    """
    output_dir = output_dir or settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or settings.batch_size

    logger.info("Loading sentence-transformer model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)

    texts = [_recipe_search_text(r) for r in recipes]
    logger.info("Encoding %d recipes (batch_size=%d) …", len(texts), batch_size)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Encoding completed in %.1f s", elapsed)

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]

    # Build flat inner-product index (cosine similarity after L2-normalisation)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", index_path, index.ntotal, dim)

    # Save the ID mapping (position → recipe_id)
    id_map = {i: r.recipe_id for i, r in enumerate(recipes)}
    id_map_path = output_dir / "faiss_id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    logger.info("ID map saved: %s", id_map_path)

    return index_path


# ---------------------------------------------------------------------------
# 2.  BM25 index
# ---------------------------------------------------------------------------
def build_bm25_index(
    recipes: List[Recipe],
    output_dir: Path | None = None,
) -> Path:
    """
    Build and pickle a BM25Okapi index.

    Returns
    -------
    Path
        Path to the pickled BM25 index.
    """
    output_dir = output_dir or settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [_recipe_search_text(r) for r in recipes]
    tokenized = [_tokenize(t) for t in texts]

    logger.info("Building BM25 index over %d documents …", len(tokenized))
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    elapsed = time.perf_counter() - t0
    logger.info("BM25 index built in %.1f s", elapsed)

    bm25_path = output_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 index saved: %s", bm25_path)

    # Also save the recipe-ID list in the same order
    bm25_ids_path = output_dir / "bm25_recipe_ids.json"
    with open(bm25_ids_path, "w", encoding="utf-8") as f:
        json.dump([r.recipe_id for r in recipes], f)
    logger.info("BM25 ID list saved: %s", bm25_ids_path)

    return bm25_path


# ---------------------------------------------------------------------------
# 3.  Inverted ingredient index
# ---------------------------------------------------------------------------
def build_inverted_index(
    recipes: List[Recipe],
    output_dir: Path | None = None,
) -> Path:
    """
    Build an inverted index: ``ingredient_name → set(recipe_ids)``.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_dir = output_dir or settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inv: Dict[str, Set[str]] = {}
    for recipe in recipes:
        for name in recipe.ingredient_names:
            if name:
                inv.setdefault(name, set()).add(recipe.recipe_id)

    # Convert sets to sorted lists for JSON serialisation
    inv_serialisable = {k: sorted(v) for k, v in sorted(inv.items())}

    inv_path = output_dir / "inverted_ingredient_index.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv_serialisable, f, ensure_ascii=False)

    logger.info(
        "Inverted index saved: %s (%d unique ingredients → %d recipe links)",
        inv_path,
        len(inv_serialisable),
        sum(len(v) for v in inv_serialisable.values()),
    )
    return inv_path


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def build_all_indices(recipes: List[Recipe] | None = None) -> Dict[str, Path]:
    """
    Build all search indices.

    Parameters
    ----------
    recipes : list[Recipe], optional
        If not provided, loads from the processed JSONL file.

    Returns
    -------
    dict[str, Path]
        Mapping of index name → file path.
    """
    if recipes is None:
        recipes = load_processed_recipes()

    paths = {}
    paths["bm25"] = build_bm25_index(recipes)
    paths["inverted"] = build_inverted_index(recipes)
    paths["faiss"] = build_faiss_index(recipes)
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    logger.info("Loading processed recipes …")
    recipes = load_processed_recipes()

    logger.info("Building all indices …")
    paths = build_all_indices(recipes)

    print("\n" + "=" * 60)
    print("INDEX BUILDING SUMMARY")
    print("=" * 60)
    for name, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name:>10s}: {path.name} ({size_mb:.1f} MB)")

    # Quick stats on inverted index
    with open(paths["inverted"], "r", encoding="utf-8") as f:
        inv = json.load(f)
    print(f"\n  Unique ingredients: {len(inv):,d}")
    top_5 = sorted(inv.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print("  Top-5 ingredients by recipe count:")
    for ing, rids in top_5:
        print(f"    {ing:30s} → {len(rids):>6,d} recipes")

    print(f"\n  FAISS vectors: {faiss.read_index(str(paths['faiss'])).ntotal:,d}")
    print("=" * 60)
