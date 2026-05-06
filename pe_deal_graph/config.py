from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-large"
EMBEDDING_DIMS: int = 1536

# ── PostgreSQL (RDS) ──────────────────────────────────────────────────
PG_HOST: str = os.environ.get("PG_HOST", "localhost")
PG_PORT: int = int(os.environ.get("PG_PORT", "5432"))
PG_USER: str = os.environ.get("PG_USER", "postgres")
PG_PASSWORD: str = os.environ.get("PG_PASSWORD", "")

LINKEDIN_DB: str = os.environ.get("LINKEDIN_DB", "linkedin_clean")

# ── Search ────────────────────────────────────────────────────────────
MAX_VECTOR_RESULTS: int = 500

# ── Valid values for structured filters ───────────────────────────────
SUPPORTED_COUNTRY_CODES = [
    "AT",
    "BE",
    "BG",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IS",
    "IT",
    "LI",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
]
