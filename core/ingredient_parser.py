"""
Ingredient string parser.

Uses regex-based extraction to split raw ingredient strings into structured
components: quantity, unit, name, and modifier.  No LLM calls – fully
deterministic.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from core.models import ParsedIngredient


# ---------------------------------------------------------------------------
# Unit vocabulary (order matters – longer tokens first to avoid partial match)
# ---------------------------------------------------------------------------
_UNITS = [
    # volume
    "tablespoons", "tablespoon", "teaspoons", "teaspoon",
    "fluid ounces", "fluid ounce", "cups", "cup",
    "gallons", "gallon", "quarts", "quart", "pints", "pint",
    "liters", "liter", "litres", "litre",
    "milliliters", "milliliter", "millilitres", "millilitre",
    "tbsps", "tbsp", "tsps", "tsp", "fl oz",
    # weight
    "pounds", "pound", "ounces", "ounce",
    "kilograms", "kilogram", "grams", "gram",
    "lbs", "lb", "oz", "kg", "g",
    # count / misc
    "cloves", "clove", "stalks", "stalk", "sprigs", "sprig",
    "slices", "slice", "heads", "head", "bunches", "bunch",
    "cans", "can", "packages", "package", "bags", "bag",
    "pinches", "pinch", "dashes", "dash", "drops", "drop",
    "pieces", "piece", "strips", "strip",
    "large", "medium", "small",
    "whole",
]

_UNIT_PATTERN = "|".join(re.escape(u) for u in _UNITS)

# ---------------------------------------------------------------------------
# Quantity pattern  –  handles: 1, 1.5, 1/2, 1 1/2, ½, 1½
# ---------------------------------------------------------------------------
_VULGAR_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8",
    "⅝": "5/8", "⅞": "7/8",
}

_FRAC_CHARS = "".join(_VULGAR_FRACTIONS.keys())
_QTY_RE = rf"(\d+\s+\d+/\d+|\d+/\d+|\d+\.?\d*|[{_FRAC_CHARS}]|\d+\s*[{_FRAC_CHARS}])"

# ---------------------------------------------------------------------------
# Modifier cues (comma-separated tail or parenthetical)
# ---------------------------------------------------------------------------
_MODIFIER_RE = re.compile(
    r",\s*(.+)$"             # everything after the last comma
    r"|"
    r"\(([^)]+)\)"           # parenthetical
)

# ---------------------------------------------------------------------------
# Full ingredient regex
# ---------------------------------------------------------------------------
_INGREDIENT_RE = re.compile(
    rf"^\s*(?P<qty>{_QTY_RE})?"       # optional quantity
    rf"\s*(?P<unit>(?:{_UNIT_PATTERN}))?"  # optional unit
    rf"\s*(?P<rest>.+)?$",            # the rest (name + modifiers)
    re.IGNORECASE,
)


def _normalise_vulgar(text: str) -> str:
    """Replace unicode vulgar fractions with ASCII equivalents."""
    for vf, ascii_frac in _VULGAR_FRACTIONS.items():
        text = text.replace(vf, ascii_frac)
    return text


def _clean_raw(raw: str) -> str:
    """Strip ADVERTISEMENT tags and normalise whitespace."""
    cleaned = re.sub(r"\s*ADVERTISEMENT\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned


def _split_name_modifier(rest: str) -> Tuple[str, Optional[str]]:
    """Separate the ingredient name from any trailing modifier."""
    if not rest:
        return "", None
    rest = rest.strip()
    # Try comma-separated modifier first
    comma_match = re.search(r",\s*(.+)$", rest)
    if comma_match:
        name = rest[: comma_match.start()].strip()
        modifier = comma_match.group(1).strip()
        return name, modifier
    # Try parenthetical modifier
    paren_match = re.search(r"\(([^)]+)\)", rest)
    if paren_match:
        name = (rest[: paren_match.start()] + rest[paren_match.end():]).strip()
        modifier = paren_match.group(1).strip()
        return name, modifier
    return rest, None


def parse_ingredient(raw: str) -> ParsedIngredient:
    """
    Parse a single raw ingredient string into a :class:`ParsedIngredient`.

    Parameters
    ----------
    raw : str
        The original ingredient string (e.g. ``"2 cups all-purpose flour, sifted"``).

    Returns
    -------
    ParsedIngredient
        Structured representation with *quantity*, *unit*, *name*, and *modifier*.
    """
    cleaned = _clean_raw(raw)
    if not cleaned:
        return ParsedIngredient(raw=raw)

    normalised = _normalise_vulgar(cleaned)
    m = _INGREDIENT_RE.match(normalised)

    if not m:
        return ParsedIngredient(raw=raw, name=cleaned)

    qty = (m.group("qty") or "").strip() or None
    unit = (m.group("unit") or "").strip() or None
    rest = (m.group("rest") or "").strip()

    name, modifier = _split_name_modifier(rest)

    # If we only got a unit but no name, the "unit" is probably part of the name
    if unit and not name:
        name = unit
        unit = None

    return ParsedIngredient(
        raw=raw,
        quantity=qty,
        unit=unit.lower() if unit else None,
        name=name.lower().strip(),
        modifier=modifier,
    )


def parse_ingredients(raw_list: list[str]) -> list[ParsedIngredient]:
    """Parse a list of raw ingredient strings."""
    return [parse_ingredient(r) for r in raw_list]
