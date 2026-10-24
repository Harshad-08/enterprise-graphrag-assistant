"""Application configuration loaded from environment variables (.env supported).

All tunable thresholds live here so they can be calibrated (see
``scripts/calibrate_thresholds.py``) and overridden without code changes.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _parse_authority_levels(raw: str) -> dict[str, int]:
    levels: dict[str, int] = {}
    for part in raw.split(","):
        if ":" in part:
            name, rank = part.split(":", 1)
            levels[name.strip().lower()] = int(rank.strip())
    return levels


def _parse_mapping(raw: str) -> dict[str, str]:
    """'a:x,b:y' -> {'a': 'x', 'b': 'y'} (values may contain '/' such as model names)."""
    mapping: dict[str, str] = {}
    for part in raw.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            mapping[key.strip()] = value.strip()
    return mapping


class Settings(BaseModel):
    # --- LLM -------------------------------------------------------------
    # "auto": Gemini if GEMINI_API_KEY is set, else Groq if GROQ_API_KEY is set, else the
    # clearly-labelled mock. "gemini"/"groq": that provider only (fails loudly without a key).
    # "mock": deterministic extractive mock.
    llm_provider: str = "auto"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "low"  # gpt-oss models: reasoning tokens count towards rate limits
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 8  # SDK retries with backoff, honouring Retry-After on HTTP 429

    # --- Neo4j / storage ---------------------------------------------------
    # "neo4j" uses the real graph database; "memory" is an in-process store with
    # the same interface, used by unit tests and offline demos.
    graph_backend: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "change-me"
    neo4j_database: str = "neo4j"

    # --- Models --------------------------------------------------------------
    embedding_backend: str = "sentence-transformers"  # or "hashing" (offline/tests)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # "nli" (local cross-encoder NLI model), "llm" (Gemini judge), "lexical" (heuristic)
    verifier_backend: str = "nli"
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    # "rule" (deterministic, default) or "llm" (Gemini adds claims; provenance-checked)
    extraction_backend: str = "rule"

    # --- Chunking ----------------------------------------------------------
    chunk_max_chars: int = 700
    chunk_overlap_sentences: int = 1

    # --- Retrieval -----------------------------------------------------------
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 6
    include_superseded_in_retrieval: bool = True

    # --- Evidence sufficiency (calibrated defaults; see calibration.json) ---
    sufficiency_threshold: float = 0.55
    min_chunk_relevance: float = 0.25
    min_concept_coverage: float = 0.5
    # The lexical concept gate is waived when the cross-encoder is this confident and all
    # query entities are covered (lexical coverage misses paraphrases like "valid" ~ "expire").
    strong_relevance: float = 0.9
    weight_relevance: float = 0.4
    weight_entity_coverage: float = 0.2
    weight_concept_coverage: float = 0.25
    weight_claim_support: float = 0.15

    # --- Recovery ----------------------------------------------------------
    recovery_enabled: bool = True
    recovery_max_attempts: int = 2
    graph_max_hops: int = 2
    graph_max_chunks: int = 30

    # --- Reliability -------------------------------------------------------
    contradiction_similarity_threshold: float = 0.75
    nli_contradiction_threshold: float = 0.7
    nli_entailment_threshold: float = 0.5
    lexical_support_threshold: float = 0.6
    max_correction_attempts: int = 1
    citation_repair_threshold: float = 0.8
    conflict_precedence: list[str] = Field(
        default_factory=lambda: ["supersession", "status", "authority", "effective_date", "version"]
    )
    authority_levels: dict[str, int] = Field(
        default_factory=lambda: {"board": 5, "executive": 4, "department": 3, "team": 2, "individual": 1}
    )

    # --- Upload --------------------------------------------------------------
    max_upload_mb: int = 10
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # --- Data / evaluation ---------------------------------------------------
    synthetic_docs_dir: Path = PROJECT_ROOT / "data" / "synthetic_documents"
    ontology_path: Path = PROJECT_ROOT / "data" / "ontology.json"
    eval_dataset_path: Path = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"
    gold_conflicts_path: Path = PROJECT_ROOT / "data" / "evaluation" / "gold_conflicts.json"
    eval_results_dir: Path = PROJECT_ROOT / "data" / "evaluation" / "results"
    calibration_path: Path = PROJECT_ROOT / "data" / "evaluation" / "calibration.json"
    ragas_enabled: bool = True
    # Judge model for Ragas. Defaults to the generation provider; on Groq a different model
    # uses a separate per-model rate-limit bucket from the one serving answers.
    ragas_judge_model: str | None = None
    # Per-metric judge override, e.g. {"context_precision": "openai/gpt-oss-20b"}
    ragas_judge_models: dict[str, str] = Field(default_factory=dict)
    ragas_max_tokens: int = 4096  # reasoning judges truncate at ragas' default of 1024
    seed_on_startup: bool = False

    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def resolved_llm_provider(self) -> str:
        if self.llm_provider == "auto":
            if self.gemini_api_key:
                return "gemini"
            return "groq" if self.groq_api_key else "mock"
        return self.llm_provider


def _apply_calibration(values: dict, calibration_path: Path) -> None:
    """Use calibrated thresholds unless the environment explicitly overrides them."""
    if not calibration_path.exists():
        return
    try:
        calibrated = json.loads(calibration_path.read_text()).get("recommended", {})
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable calibration file %s: %s", calibration_path, exc)
        return
    for key in ("sufficiency_threshold", "min_chunk_relevance", "min_concept_coverage", "strong_relevance"):
        if key in calibrated and os.getenv(key.upper()) in (None, ""):
            values[key] = float(calibrated[key])


def load_settings() -> Settings:
    defaults = Settings()
    values: dict = {
        "llm_provider": _env("LLM_PROVIDER", defaults.llm_provider).lower(),
        "gemini_api_key": os.getenv("GEMINI_API_KEY") or None,
        "gemini_model": _env("GEMINI_MODEL", defaults.gemini_model),
        "groq_api_key": os.getenv("GROQ_API_KEY") or None,
        "groq_model": _env("GROQ_MODEL", defaults.groq_model),
        "groq_reasoning_effort": _env("GROQ_REASONING_EFFORT", defaults.groq_reasoning_effort).lower(),
        "groq_base_url": _env("GROQ_BASE_URL", defaults.groq_base_url),
        "llm_max_retries": _env_int("LLM_MAX_RETRIES", defaults.llm_max_retries),
        "ragas_judge_model": os.getenv("RAGAS_JUDGE_MODEL") or None,
        "ragas_judge_models": _parse_mapping(os.getenv("RAGAS_JUDGE_MODELS", "")),
        "ragas_max_tokens": _env_int("RAGAS_MAX_TOKENS", defaults.ragas_max_tokens),
        "llm_temperature": _env_float("LLM_TEMPERATURE", defaults.llm_temperature),
        "llm_timeout_seconds": _env_int("LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds),
        "graph_backend": _env("GRAPH_BACKEND", defaults.graph_backend).lower(),
        "neo4j_uri": _env("NEO4J_URI", defaults.neo4j_uri),
        "neo4j_username": _env("NEO4J_USERNAME", defaults.neo4j_username),
        "neo4j_password": _env("NEO4J_PASSWORD", defaults.neo4j_password),
        "neo4j_database": _env("NEO4J_DATABASE", defaults.neo4j_database),
        "embedding_backend": _env("EMBEDDING_BACKEND", defaults.embedding_backend).lower(),
        "embedding_model": _env("EMBEDDING_MODEL", defaults.embedding_model),
        "embedding_dimension": _env_int("EMBEDDING_DIMENSION", defaults.embedding_dimension),
        "reranker_enabled": _env_bool("RERANKER_ENABLED", defaults.reranker_enabled),
        "reranker_model": _env("RERANKER_MODEL", defaults.reranker_model),
        "verifier_backend": _env("VERIFIER_BACKEND", defaults.verifier_backend).lower(),
        "nli_model": _env("NLI_MODEL", defaults.nli_model),
        "extraction_backend": _env("EXTRACTION_BACKEND", defaults.extraction_backend).lower(),
        "chunk_max_chars": _env_int("CHUNK_MAX_CHARS", defaults.chunk_max_chars),
        "chunk_overlap_sentences": _env_int("CHUNK_OVERLAP_SENTENCES", defaults.chunk_overlap_sentences),
        "vector_top_k": _env_int("VECTOR_TOP_K", defaults.vector_top_k),
        "keyword_top_k": _env_int("KEYWORD_TOP_K", defaults.keyword_top_k),
        "rrf_k": _env_int("RRF_K", defaults.rrf_k),
        "rerank_candidates": _env_int("RERANK_CANDIDATES", defaults.rerank_candidates),
        "final_top_k": _env_int("FINAL_TOP_K", defaults.final_top_k),
        "include_superseded_in_retrieval": _env_bool(
            "INCLUDE_SUPERSEDED_IN_RETRIEVAL", defaults.include_superseded_in_retrieval
        ),
        "sufficiency_threshold": _env_float("SUFFICIENCY_THRESHOLD", defaults.sufficiency_threshold),
        "min_chunk_relevance": _env_float("MIN_CHUNK_RELEVANCE", defaults.min_chunk_relevance),
        "min_concept_coverage": _env_float("MIN_CONCEPT_COVERAGE", defaults.min_concept_coverage),
        "strong_relevance": _env_float("STRONG_RELEVANCE", defaults.strong_relevance),
        "recovery_enabled": _env_bool("RECOVERY_ENABLED", defaults.recovery_enabled),
        "recovery_max_attempts": _env_int("RECOVERY_MAX_ATTEMPTS", defaults.recovery_max_attempts),
        "graph_max_hops": _env_int("GRAPH_MAX_HOPS", defaults.graph_max_hops),
        "graph_max_chunks": _env_int("GRAPH_MAX_CHUNKS", defaults.graph_max_chunks),
        "contradiction_similarity_threshold": _env_float(
            "CONTRADICTION_SIMILARITY_THRESHOLD", defaults.contradiction_similarity_threshold
        ),
        "nli_contradiction_threshold": _env_float(
            "NLI_CONTRADICTION_THRESHOLD", defaults.nli_contradiction_threshold
        ),
        "nli_entailment_threshold": _env_float("NLI_ENTAILMENT_THRESHOLD", defaults.nli_entailment_threshold),
        "lexical_support_threshold": _env_float(
            "LEXICAL_SUPPORT_THRESHOLD", defaults.lexical_support_threshold
        ),
        "max_correction_attempts": _env_int("MAX_CORRECTION_ATTEMPTS", defaults.max_correction_attempts),
        "citation_repair_threshold": _env_float("CITATION_REPAIR_THRESHOLD", defaults.citation_repair_threshold),
        "max_upload_mb": _env_int("MAX_UPLOAD_MB", defaults.max_upload_mb),
        "ragas_enabled": _env_bool("RAGAS_ENABLED", defaults.ragas_enabled),
        "seed_on_startup": _env_bool("SEED_ON_STARTUP", defaults.seed_on_startup),
        "log_level": _env("LOG_LEVEL", defaults.log_level).upper(),
    }
    precedence = os.getenv("CONFLICT_PRECEDENCE")
    if precedence:
        values["conflict_precedence"] = [p.strip() for p in precedence.split(",") if p.strip()]
    authority = os.getenv("AUTHORITY_LEVELS")
    if authority:
        values["authority_levels"] = _parse_authority_levels(authority)
    if os.getenv("UPLOAD_DIR"):
        values["upload_dir"] = Path(os.environ["UPLOAD_DIR"])
    _apply_calibration(values, defaults.calibration_path)
    return Settings(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
