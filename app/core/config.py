from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str | None = "low"
    google_api_key: SecretStr
    gemini_embedding_model: str = "gemini-embedding-001"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "policies"
    retrieval_k: int = 4
    retrieval_min_score: float = 0.6

    data_dir: Path = ROOT / "data"
    checkpoint_db: Path = ROOT / "var" / "checkpoints.sqlite"

    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "curly-engine"
    tracing_enabled: bool = True

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def orders_file(self) -> Path:
        return self.data_dir / "orders.md"

    @property
    def returns_file(self) -> Path:
        return self.data_dir / "returns.md"


@lru_cache
def get_settings() -> Settings:
    return Settings()
