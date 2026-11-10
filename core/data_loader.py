"""
Data loader for the Recipe Agent.

Loads all three raw JSON recipe files, assigns stable IDs, cleans data,
parses ingredients, and persists processed results to ``data/processed/``.

Stable ID format:
  - ``ar_00001`` … ``ar_NNNNN``  (AllRecipes)
  - ``epi_00001`` … ``epi_NNNNN`` (Epicurious)
  - ``fn_00001`` … ``fn_NNNNN``  (FoodNetwork)
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from config.settings import settings
from core.ingredient_parser import parse_ingredients
from core.models import Recipe
from core.normalizer import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source file definitions
# ---------------------------------------------------------------------------
_SOURCE_FILES: List[Tuple[str, str]] = [
    ("ar", "recipes_raw_nosource_ar.json"),
    ("epi", "recipes_raw_nosource_epi.json"),
    ("fn", "recipes_raw_nosource_fn.json"),
]


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------
def _strip_advertisement(text: str) -> str:
    """Remove ADVERTISEMENT tags (standalone or inline)."""
    return re.sub(r"\s*ADVERTISEMENT\s*", " ", text, flags=re.IGNORECASE).strip()


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form and replace common garbled sequences."""
    text = unicodedata.normalize("NFC", text)
    # Common Mojibake repairs (latin-1 interpreted as utf-8)
    text = text.replace("Â\u00b0", "°")
    text = text.replace("\u00c2\u00b0", "°")
    text = text.replace("â\u0080\u0093", "–")
    text = text.replace("â\u0080\u0094", "—")
    text = text.replace("â\u0080\u0099", "'")
    text = text.replace("â\u0080\u009c", '"')
    text = text.replace("â\u0080\u009d", '"')
    return text


def _split_instructions(instructions: str) -> List[str]:
    """
    Split an instruction string into individual steps.

    Handles:
      - Newline-delimited steps
      - Numbered steps (``1.``, ``2.``, ``Step 1:``, etc.)
    """
    if not instructions:
        return []

    # Try splitting on newlines first
    lines = [line.strip() for line in instructions.split("\n") if line.strip()]

    # If only one block, try numbered pattern
    if len(lines) == 1:
        numbered = re.split(r"(?:^|\.\s+)(?=\d+[\.\)]\s)", lines[0])
        numbered = [s.strip() for s in numbered if s.strip()]
        if len(numbered) > 1:
            return numbered

    return lines


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------
def _load_source(source_prefix: str, filename: str) -> List[Recipe]:
    """Load a single JSON source file and return a list of Recipe objects."""
    filepath = settings.raw_data_dir / filename
    logger.info("Loading %s from %s …", source_prefix.upper(), filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data: Dict = json.load(f)

    recipes: List[Recipe] = []
    for idx, (hash_key, entry) in enumerate(sorted(raw_data.items()), start=1):
        recipe_id = f"{source_prefix}_{idx:05d}"

        title = _normalize_unicode(_strip_advertisement(entry.get("title", "") or "")).strip()

        # Clean ingredient strings
        raw_ingredients = entry.get("ingredients", []) or []
        cleaned_ingredients = []
        for ing in raw_ingredients:
            cleaned = _strip_advertisement(_normalize_unicode(ing or ""))
            if cleaned:  # drop empty / ADVERTISEMENT-only entries
                cleaned_ingredients.append(cleaned)

        # Parse ingredients into structured form
        parsed = parse_ingredients(cleaned_ingredients)

        # Normalize ingredient names
        for p in parsed:
            p.normalized_name = normalize(p.name)

        ingredient_names = [p.normalized_name for p in parsed if p.normalized_name]

        # Instructions
        raw_instructions = _normalize_unicode(
            _strip_advertisement(entry.get("instructions", "") or "")
        )
        steps = _split_instructions(raw_instructions)

        recipe = Recipe(
            recipe_id=recipe_id,
            source=source_prefix,
            title=title,
            ingredients_raw=cleaned_ingredients,
            ingredients_parsed=parsed,
            instructions=raw_instructions,
            instruction_steps=steps,
            picture_link=entry.get("picture_link"),
            original_key=hash_key,
            ingredient_names=ingredient_names,
        )
        recipes.append(recipe)

    logger.info("  → Loaded %d recipes from %s", len(recipes), source_prefix.upper())
    return recipes


def ensure_raw_data():
    """Ensure raw dataset files exist, downloading and unzipping if missing."""
    import urllib.request
    import zipfile
    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_exist = all((raw_dir / fn).exists() for _, fn in _SOURCE_FILES)
    if all_exist:
        return
    zip_path = raw_dir / "recipes_raw.zip"
    if not zip_path.exists():
        url = "https://eightportions.com/datasets/Recipes/recipes_raw.zip"
        logger.info("Downloading dataset from %s ...", url)
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)
    logger.info("Raw dataset extracted successfully.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_all_recipes() -> List[Recipe]:
    """
    Load and process all recipes from the three source JSON files.

    Returns
    -------
    list[Recipe]
        Fully processed recipe objects with stable IDs, parsed ingredients,
        and split instruction steps.
    """
    ensure_raw_data()
    all_recipes: List[Recipe] = []
    for prefix, filename in _SOURCE_FILES:
        all_recipes.extend(_load_source(prefix, filename))
    logger.info("Total recipes loaded: %d", len(all_recipes))
    return all_recipes


def save_processed_recipes(recipes: List[Recipe], output_dir: Path | None = None) -> Path:
    """
    Persist processed recipes to a JSON-lines file.

    Parameters
    ----------
    recipes : list[Recipe]
        The recipes to save.
    output_dir : Path, optional
        Directory to write to; defaults to ``data/processed/``.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir = output_dir or settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "recipes.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for r in recipes:
            f.write(r.to_json() + "\n")

    logger.info("Saved %d recipes to %s", len(recipes), output_path)
    return output_path


def load_processed_recipes(input_path: Path | None = None) -> List[Recipe]:
    """Load recipes from cached pickle if available, otherwise JSONL."""
    pkl_path = settings.processed_data_dir / "recipes.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            recipes = pickle.load(f)
        logger.info("Loaded %d recipes from pickle %s", len(recipes), pkl_path)
        return recipes

    input_path = input_path or (settings.processed_data_dir / "recipes.jsonl")
    recipes = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recipes.append(Recipe.from_json(line))
    logger.info("Loaded %d processed recipes from %s", len(recipes), input_path)

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(recipes, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached recipes to %s for ultra-fast boot", pkl_path)
    except Exception as e:
        logger.warning("Could not write pickle cache: %s", e)

    return recipes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    recipes = load_all_recipes()
    path = save_processed_recipes(recipes)

    # Print summary statistics
    sources = {}
    for r in recipes:
        sources[r.source] = sources.get(r.source, 0) + 1

    print("\n" + "=" * 60)
    print("DATA LOADING SUMMARY")
    print("=" * 60)
    for src, count in sorted(sources.items()):
        print(f"  {src.upper():>4s}: {count:>6,d} recipes")
    print(f"  {'TOTAL':>4s}: {len(recipes):>6,d} recipes")
    print(f"\n  Output: {path}")

    # Sample
    sample = recipes[0]
    print(f"\n  Sample recipe: {sample.recipe_id}")
    print(f"    Title: {sample.title}")
    print(f"    Ingredients ({len(sample.ingredients_parsed)}):")
    for ip in sample.ingredients_parsed[:3]:
        print(f"      qty={ip.quantity!r}  unit={ip.unit!r}  name={ip.name!r}  mod={ip.modifier!r}  norm={ip.normalized_name!r}")
    print(f"    Steps: {len(sample.instruction_steps)}")
    print("=" * 60)
