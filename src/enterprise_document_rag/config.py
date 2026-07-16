import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration kept explicit so later tasks can swap local backends."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = "sqlite:///./data/agent.db"
    vector_backend: str = "qdrant_local"
    qdrant_path: Path = Path("./data/qdrant")
    vector_collection_name: str = "document_chunks_bge_small_zh_v1_5"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_backend: str = "bge"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=8, ge=1, le=64)
    reranker_enabled: bool = False
    reranker_backend: str = "bge"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=16, ge=1, le=64)
    llm_backend: str = "qwen_transformers"
    llm_model_id: str = "Qwen/Qwen3-0.6B"
    llm_max_new_tokens: int = Field(default=320, ge=32, le=512)
    huggingface_home: Path | None = None
    authorized_roots: str = "./knowledge"
    chunk_size_tokens: int = Field(default=420, ge=1)
    chunk_overlap_tokens: int = Field(default=40, ge=0)
    chunk_min_tokens: int = Field(default=180, ge=1)
    chunk_max_tokens: int = Field(default=650, ge=1)
    ocr_enabled: bool = True
    ocr_min_text_chars_per_page: int = Field(default=40, ge=0)
    ocr_render_dpi: int = Field(default=150, ge=72, le=300)
    archive_max_members: int = Field(default=500, ge=1, le=10_000)
    archive_max_member_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    archive_max_uncompressed_bytes: int = Field(default=200 * 1024 * 1024, ge=1)
    archive_max_compression_ratio: int = Field(default=100, ge=1, le=1_000)
    vector_top_k: int = Field(default=40, ge=1)
    fts_top_k: int = Field(default=40, ge=1)
    retrieval_candidate_top_k: int = Field(default=12, ge=1)
    max_chunks_per_document: int = Field(default=1, ge=1)
    final_top_k: int = Field(default=10, ge=1)
    graph_rag_enabled: bool = False
    app_data_dir: Path | None = None
    update_enabled: bool = True
    update_repository: str = "ykykk000009/enterprise-rag"
    update_check_interval_hours: int = Field(default=24, ge=1, le=168)
    update_request_timeout_seconds: int = Field(default=15, ge=3, le=120)

    @property
    def authorized_root_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(item.strip()) for item in self.authorized_roots.split(",") if item.strip()
        )

    @property
    def database_path(self) -> Path:
        if self.database_url == "sqlite:///:memory:":
            return Path(":memory:")
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.removeprefix("sqlite:///"))
        return Path(self.database_url)

    @property
    def application_data_path(self) -> Path:
        if self.app_data_dir is not None:
            return self.app_data_dir.expanduser().resolve()
        database = self.database_path
        if database == Path(":memory:"):
            return Path.cwd()
        resolved = database.expanduser().resolve()
        return resolved.parent.parent if resolved.parent.name.lower() == "data" else resolved.parent


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_huggingface_cache(settings: Settings) -> None:
    """Keep downloaded local models in the configured workspace cache."""
    if settings.huggingface_home is None:
        return
    home = settings.huggingface_home.expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(home / "hub")
