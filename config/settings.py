"""
Pydantic settings for the Recipe Agent project.

Loads configuration from .env file with support for multiple Groq API keys,
model selection, and data/processing paths.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ----- Groq API keys (round-robin pool) -----
    groq_api_key_1: str = Field(default="", description="Primary Groq API key")
    groq_api_key_2: str = Field(default="", description="Secondary Groq API key")
    groq_api_key_3: str = Field(default="", description="Tertiary Groq API key")

    # ----- Model selection -----
    primary_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Primary LLM model identifier on Groq",
    )
    fallback_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Fallback LLM model identifier on Groq",
    )

    # ----- Embedding model -----
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model for FAISS embeddings",
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimensionality of the embedding model output",
    )

    # ----- Processing tunables -----
    batch_size: int = Field(default=512, description="Batch size for embedding")
    faiss_nlist: int = Field(
        default=100,
        description="Number of Voronoi cells for IVF index (ignored for flat index)",
    )

    # ----- Paths (derived, not from env) -----
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR

    # ----- Helpers -----
    @property
    def groq_api_keys(self) -> List[str]:
        """Return a list of all non-empty Groq API keys."""
        return [
            k
            for k in [self.groq_api_key_1, self.groq_api_key_2, self.groq_api_key_3]
            if k
        ]

    @property
    def active_groq_key(self) -> Optional[str]:
        """Return the first available Groq API key, or None."""
        keys = self.groq_api_keys
        return keys[0] if keys else None


# Singleton instance
settings = Settings()
