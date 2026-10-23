from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # LLM & Embeddings Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # Server Configuration
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # Storage & Knowledge Paths
    DATA_DIR: Path = BASE_DIR / "data"
    MOCK_CUSTOMERS_PATH: Path = BASE_DIR / "data" / "mock_customers.json"
    CORPUS_JSON_PATH: Path = BASE_DIR / "data" / "getnet_corpus.json"
    VECTOR_INDEX_PATH: Path = BASE_DIR / "data" / "vector_index.json"
    EVAL_DATASET_PATH: Path = BASE_DIR / "data" / "eval_dataset.json"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "getnet_state.db"

    # Orchestration & RAG Thresholds
    MAX_ORCHESTRATION_ITERATIONS: int = 4
    MAX_TOOL_TURNS: int = 4
    TOOL_TIMEOUT_SECONDS: float = 5.0
    RAG_MIN_RELEVANCE_THRESHOLD: float = 0.26
    RAG_TOP_K: int = 3
    MEMORY_WINDOW_MESSAGES: int = 8

    # Security, Auth & Rate Limiting
    AUTH_API_KEY: str = "getnet-fde-secret-key-2026"
    JWT_SECRET: str = "getnet-hmac-jwt-signing-secret-2026"
    AUTH_ENFORCE_STRICT: bool = False
    RATE_LIMIT_PER_MINUTE: int = 120
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000",
    ]

    # Resilience (Circuit Breaker & Retries)
    CB_FAILURE_THRESHOLD: int = 3
    CB_RECOVERY_TIMEOUT_SECONDS: float = 30.0
    HTTP_MAX_RETRIES: int = 2
    HTTP_TIMEOUT_SECONDS: float = 4.0

    # Optional Human-in-the-Loop (HITL) External Channels
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    HITL_WEBHOOK_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
