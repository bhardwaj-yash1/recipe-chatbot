"""
LLM Client — Groq/LangChain wrapper with round-robin key rotation.

Provides a single call interface that automatically:
  - Rotates across multiple API keys
  - Retries on rate-limit (429) with exponential backoff
  - Falls back to a smaller model when the primary is unavailable
  - Enforces JSON mode for structured outputs
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Round-robin Groq LLM client with resilience."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        self._keys = api_keys or settings.groq_api_keys
        if not self._keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEY_1/2/3 in .env")

        self._primary = primary_model or settings.primary_model
        self._fallback = fallback_model or settings.fallback_model
        self._key_index = 0
        self._call_count = 0

    def _next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _make_llm(self, model: str, api_key: str, json_mode: bool = False) -> ChatGroq:
        kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**kwargs)

    def invoke(
        self,
        system_prompt: str,
        user_message: str,
        json_mode: bool = False,
        max_retries: int = 6,
    ) -> str:
        """
        Send a message and return the LLM's text response.

        Tries all API keys with exponential backoff before falling back
        to the smaller model.
        """
        models_to_try = [self._primary, self._fallback]
        last_error = None

        for model in models_to_try:
            for attempt in range(max_retries):
                api_key = self._next_key()
                try:
                    llm = self._make_llm(model, api_key, json_mode=json_mode)
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_message),
                    ]
                    response = llm.invoke(messages)
                    self._call_count += 1
                    return response.content

                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()

                    if "429" in err_str or "rate" in err_str:
                        wait = min(2 ** attempt, 30)
                        logger.warning(
                            "Rate limited (key #%d, model=%s). Retrying in %ds…",
                            self._key_index, model, wait,
                        )
                        time.sleep(wait)
                        continue

                    elif "model" in err_str and ("not found" in err_str or "not available" in err_str):
                        logger.warning("Model %s not available, trying fallback…", model)
                        break  # Skip to next model

                    else:
                        logger.error("LLM error: %s", e)
                        wait = min(2 ** attempt, 15)
                        time.sleep(wait)
                        continue

        raise RuntimeError(f"All LLM attempts exhausted. Last error: {last_error}")

    def invoke_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Dict:
        """Invoke with JSON mode and parse the response."""
        raw = self.invoke(system_prompt, user_message, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                return json.loads(raw[start:end].strip())
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                return json.loads(raw[start:end].strip())
            # Try to find JSON object in the text
            for i, ch in enumerate(raw):
                if ch == "{":
                    try:
                        return json.loads(raw[i:])
                    except json.JSONDecodeError:
                        continue
            logger.error("Failed to parse JSON from LLM response: %s", raw[:500])
            return {"error": "Failed to parse response", "raw": raw}

    @property
    def call_count(self) -> int:
        return self._call_count
