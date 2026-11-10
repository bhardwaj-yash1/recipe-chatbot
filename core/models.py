"""
Data models for the Recipe Agent.

Uses Python dataclasses (serialisable to/from dict/JSON) to keep
the domain layer free of heavy framework dependencies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ParsedIngredient
# ---------------------------------------------------------------------------
@dataclass
class ParsedIngredient:
    """A single ingredient string parsed into structured components."""

    raw: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    name: str = ""
    modifier: Optional[str] = None
    normalized_name: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ParsedIngredient":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------
@dataclass
class Recipe:
    """Fully processed recipe record with a stable ID."""

    recipe_id: str                              # e.g. ar_00001
    source: str                                 # ar | epi | fn
    title: str = ""
    ingredients_raw: List[str] = field(default_factory=list)
    ingredients_parsed: List[ParsedIngredient] = field(default_factory=list)
    instructions: str = ""
    instruction_steps: List[str] = field(default_factory=list)
    picture_link: Optional[str] = None
    original_key: str = ""                      # hash key from raw JSON

    # --- Derived search helpers (populated during indexing) ---
    ingredient_names: List[str] = field(default_factory=list)

    # ----- Serialization ---------------------------------------------------
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["ingredients_parsed"] = [ip.to_dict() for ip in self.ingredients_parsed]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Recipe":
        parsed = [
            ParsedIngredient.from_dict(p) for p in d.pop("ingredients_parsed", [])
        ]
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**filtered, ingredients_parsed=parsed)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Recipe":
        return cls.from_dict(json.loads(s))
