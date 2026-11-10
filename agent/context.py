"""
Conversation Context Manager — tracks multi-turn state.

This is NOT fuzzy "LLM memory". It is an explicit, deterministic
state container that tracks active search results, filters,
matching context, and recently referenced recipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.intents import Intent


@dataclass
class Turn:
    """A single conversation turn."""
    user_message: str
    intent: Optional[Intent] = None
    params: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    result_data: Any = None


@dataclass
class MatchContext:
    """State for an active ingredient-matching session."""
    available: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    max_steps: Optional[int] = None
    prefer: str = "fewest_missing"


class ConversationContext:
    """Deterministic conversation state tracker."""

    def __init__(self):
        self.history: List[Turn] = []
        self.active_search_results: List[str] = []  # recipe IDs
        self.active_filters: Dict[str, Any] = {}
        self.active_match_context: Optional[MatchContext] = None
        self.last_intent: Optional[Intent] = None
        self.mentioned_recipes: Dict[str, Any] = {}  # id -> recipe data
        self.last_recipe_id: Optional[str] = None

    def add_turn(self, turn: Turn):
        self.history.append(turn)
        if turn.intent:
            self.last_intent = turn.intent

    def set_search_results(self, recipe_ids: List[str]):
        self.active_search_results = recipe_ids

    def set_match_context(self, ctx: MatchContext):
        self.active_match_context = ctx

    def set_last_recipe(self, recipe_id: str):
        self.last_recipe_id = recipe_id

    def add_filter(self, key: str, value: Any):
        if key == "exclude_ingredients":
            existing = self.active_filters.get(key, [])
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
            self.active_filters[key] = existing
        else:
            self.active_filters[key] = value

    def get_context_summary(self) -> str:
        """Build a text summary of the conversation state for prompt injection."""
        parts = []

        if self.history:
            recent = self.history[-5:]
            parts.append("Recent conversation:")
            for t in recent:
                parts.append(f"  User: {t.user_message}")
                if t.intent:
                    parts.append(f"  Intent: {t.intent.value}")
                if t.response:
                    parts.append(f"  Bot: {t.response[:200]}...")

        if self.active_search_results:
            parts.append(f"\nActive search results: {len(self.active_search_results)} recipe(s)")
            parts.append(f"  IDs: {', '.join(self.active_search_results[:10])}")

        if self.active_filters:
            parts.append(f"\nActive filters: {self.active_filters}")

        if self.active_match_context:
            mc = self.active_match_context
            parts.append(f"\nActive matching session:")
            parts.append(f"  Available: {mc.available}")
            parts.append(f"  Excluded: {mc.excluded}")
            if mc.max_steps:
                parts.append(f"  Max steps: {mc.max_steps}")

        if self.last_recipe_id:
            parts.append(f"\nLast referenced recipe: {self.last_recipe_id}")

        if self.last_intent:
            parts.append(f"\nLast intent: {self.last_intent.value}")

        return "\n".join(parts) if parts else "No prior context."

    def get_user_messages(self) -> List[str]:
        """Return all user messages (for hallucination guard)."""
        return [t.user_message for t in self.history]

    def clear(self):
        self.__init__()
