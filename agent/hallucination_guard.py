"""
Hallucination Guard — scrubs extracted parameters that cannot be traced
back to the user's input or conversation context.

This is the critical safety net from the reference architecture: if the
LLM invents a parameter value (e.g. returns "4 servings" when the user
never mentioned servings), the guard deletes it before execution.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Values that are always acceptable defaults
_SAFE_DEFAULTS = {
    "limit", "top_n", "ascending", "mode", "prefer",
}

# Short generic values that are OK even without exact match
_GENERIC_VALUES = {"all", "any", "true", "false", "yes", "no"}


def scrub_hallucinated_params(
    extracted: Dict[str, Any],
    user_message: str,
    context_messages: List[str],
    intent_name: str = "",
) -> Dict[str, Any]:
    """
    Remove any extracted parameter whose value cannot be found in the
    user's message or recent conversation context.

    Parameters
    ----------
    extracted : dict
        Parameters extracted by the LLM.
    user_message : str
        The current user message.
    context_messages : list[str]
        Recent user messages from conversation history.
    intent_name : str
        Name of the classified intent (for logging).

    Returns
    -------
    dict
        Cleaned parameters with hallucinated values removed.
    """
    # Build the full text corpus to search against
    all_text = (user_message + " " + " ".join(context_messages)).lower()
    # Also build a version without common stopwords for fuzzy matching
    all_words = set(re.findall(r'\b\w+\b', all_text))

    cleaned: Dict[str, Any] = {}

    for key, value in extracted.items():
        # Skip None values
        if value is None:
            continue

        # Safe default params always pass
        if key in _SAFE_DEFAULTS:
            cleaned[key] = value
            continue

        # Check based on value type
        if isinstance(value, bool):
            cleaned[key] = value
            continue

        if isinstance(value, (int, float)):
            # Numbers: check if the number appears in the text
            if str(int(value)) in all_text or str(value) in all_text:
                cleaned[key] = value
            elif key in _SAFE_DEFAULTS:
                cleaned[key] = value
            else:
                logger.warning(
                    "HALLUCINATION SCRUB [%s]: Removed %s=%s (number not in user input)",
                    intent_name, key, value,
                )
            continue

        if isinstance(value, str):
            val_lower = value.lower().strip()
            # Generic values are always OK
            if val_lower in _GENERIC_VALUES:
                cleaned[key] = value
                continue
            # Check if value (or significant words from it) appear in context
            val_words = set(re.findall(r'\b\w+\b', val_lower))
            # At least 50% of the value's words must appear in the user text
            if val_words:
                overlap = val_words & all_words
                ratio = len(overlap) / len(val_words)
                if ratio >= 0.5 or val_lower in all_text:
                    cleaned[key] = value
                else:
                    logger.warning(
                        "HALLUCINATION SCRUB [%s]: Removed %s='%s' (%.0f%% word overlap, need 50%%)",
                        intent_name, key, value, ratio * 100,
                    )
            else:
                cleaned[key] = value  # Empty string, let it through
            continue

        if isinstance(value, list):
            # For lists, scrub each element individually
            clean_list = []
            for item in value:
                if isinstance(item, str):
                    item_lower = item.lower().strip()
                    item_words = set(re.findall(r'\b\w+\b', item_lower))
                    if item_words:
                        overlap = item_words & all_words
                        ratio = len(overlap) / len(item_words)
                        if ratio >= 0.5 or item_lower in all_text:
                            clean_list.append(item)
                        else:
                            logger.warning(
                                "HALLUCINATION SCRUB [%s]: Removed list item '%s' from %s",
                                intent_name, item, key,
                            )
                    else:
                        clean_list.append(item)
                else:
                    clean_list.append(item)
            if clean_list:
                cleaned[key] = clean_list
            continue

        # Other types pass through
        cleaned[key] = value

    return cleaned
