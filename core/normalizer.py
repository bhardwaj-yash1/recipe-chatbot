"""
Ingredient name normalizer.

Provides:
  - Synonym mapping (e.g. "scallion" → "green onion")
  - Lemmatization via NLTK WordNet
  - Lowercasing, whitespace collapse
  - A simple pipeline: ``normalize(name) -> canonical_name``
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict

import nltk

# Ensure WordNet data is available
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Synonym map  (target ← aliases)
# ---------------------------------------------------------------------------
SYNONYM_MAP: Dict[str, str] = {
    # Alliums
    "green onion": "scallion",
    "green onion": "spring onion",
    # Peppers
    "bell pepper": "capsicum",
    "chili pepper": "chilli pepper",
    "chili pepper": "chile pepper",
    "chili": "chilli",
    # Dairy
    "heavy cream": "heavy whipping cream",
    "heavy cream": "whipping cream",
    "cream cheese": "philadelphia cream cheese",
    # Herbs
    "cilantro": "coriander leaves",
    "cilantro": "fresh coriander",
    # Proteins
    "ground beef": "minced beef",
    "ground beef": "hamburger meat",
    "chicken breast": "boneless skinless chicken breast",
    "chicken breast": "skinless boneless chicken breast",
    # Staples
    "all-purpose flour": "all purpose flour",
    "all-purpose flour": "plain flour",
    "powdered sugar": "confectioners sugar",
    "powdered sugar": "icing sugar",
    "baking soda": "bicarbonate of soda",
    # Oils
    "vegetable oil": "canola oil",
    "olive oil": "extra virgin olive oil",
    "olive oil": "extra-virgin olive oil",
    # Canned
    "canned tomatoes": "tinned tomatoes",
    "tomato paste": "tomato puree",
    # Misc
    "cornstarch": "corn starch",
    "soy sauce": "soya sauce",
    "eggplant": "aubergine",
    "zucchini": "courgette",
    "arugula": "rocket",
    "garbanzo beans": "chickpeas",
    "garbanzo beans": "chick peas",
}

# Build reverse lookup: alias -> canonical
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, alias in SYNONYM_MAP.items():
    _ALIAS_TO_CANONICAL[alias.lower()] = canonical.lower()
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical.lower()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
_NOISE_WORDS = re.compile(
    r"\b(fresh|dried|ground|chopped|minced|diced|sliced|crushed|"
    r"shredded|grated|packed|loosely|firmly|about|approximately|"
    r"finely|coarsely|thinly|thickly|roughly|to taste|optional|"
    r"garnish|for serving|for garnish)\b",
    re.IGNORECASE,
)

_WHITESPACE = re.compile(r"\s+")


def _remove_noise(text: str) -> str:
    """Remove common modifiers / noise words for canonical comparison."""
    return _NOISE_WORDS.sub("", text).strip()


@lru_cache(maxsize=32_768)
def _lemmatize_word(word: str) -> str:
    """Lemmatize a single word (noun)."""
    return _lemmatizer.lemmatize(word, pos="n")


def lemmatize_phrase(phrase: str) -> str:
    """Lemmatize every word in a phrase."""
    return " ".join(_lemmatize_word(w) for w in phrase.split())


def normalize(name: str, *, strip_noise: bool = False) -> str:
    """
    Normalise an ingredient name to a canonical form.

    Steps:
      1. Lowercase
      2. Collapse whitespace
      3. (Optional) Remove noise/modifier words
      4. Apply synonym mapping
      5. Lemmatize each word

    Parameters
    ----------
    name : str
        Raw or semi-parsed ingredient name.
    strip_noise : bool
        If True, strip common modifier words before canonicalising.

    Returns
    -------
    str
        The canonical ingredient name.
    """
    if not name:
        return ""
    text = name.lower().strip()
    text = _WHITESPACE.sub(" ", text)
    if strip_noise:
        text = _remove_noise(text)
        text = _WHITESPACE.sub(" ", text).strip()

    # Synonym lookup (check full phrase first)
    text = _ALIAS_TO_CANONICAL.get(text, text)

    # Lemmatize
    text = lemmatize_phrase(text)
    return text


def normalize_ingredient_list(names: list[str], *, strip_noise: bool = False) -> list[str]:
    """Normalise a list of ingredient names."""
    return [normalize(n, strip_noise=strip_noise) for n in names]
