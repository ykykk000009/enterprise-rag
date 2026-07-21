import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import Settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    embedding_model TEXT NOT NULL,
    vector_backend TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    root_path TEXT NOT NULL,
    include_patterns TEXT NOT NULL DEFAULT '[]',
    exclude_patterns TEXT NOT NULL DEFAULT '[]',
    scan_state TEXT NOT NULL DEFAULT 'idle',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (knowledge_base_id, root_path)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    canonical_path TEXT NOT NULL,
    active_version_id TEXT,
    visibility_state TEXT NOT NULL DEFAULT 'visible',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (knowledge_base_id, canonical_path)
);

CREATE TABLE IF NOT EXISTS document_versions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mtime_ns INTEGER NOT NULL CHECK (mtime_ns >= 0),
    sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    layout_version TEXT,
    document_type TEXT,
    document_structure_tree TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, sha256)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_version_id TEXT NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    page_no INTEGER,
    page_range TEXT,
    section_path TEXT,
    bbox TEXT,
    bbox_list TEXT NOT NULL DEFAULT '[]',
    content_type TEXT NOT NULL DEFAULT 'text',
    source_type TEXT NOT NULL DEFAULT 'native_text',
    ocr_confidence REAL,
    block_types TEXT NOT NULL DEFAULT '[]',
    table_markdown TEXT,
    image_path TEXT,
    caption TEXT,
    image_metadata TEXT,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    text_hash TEXT NOT NULL,
    previous_chunk_id TEXT,
    next_chunk_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_version_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS index_records (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    index_kind TEXT NOT NULL,
    external_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (chunk_id, index_kind, index_version)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_version_id UNINDEXED,
    text
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    job_key TEXT NOT NULL UNIQUE,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    path TEXT NOT NULL,
    expected_sha256 TEXT,
    state TEXT NOT NULL DEFAULT 'queued',
    payload TEXT NOT NULL DEFAULT '{}',
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    lease_owner TEXT,
    lease_expires_at TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS roles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS permissions (
    id TEXT PRIMARY KEY,
    role_id TEXT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    permission TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (role_id, knowledge_base_id, permission)
);

CREATE TABLE IF NOT EXISTS query_audits (
    id TEXT PRIMARY KEY,
    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    filters TEXT NOT NULL DEFAULT '{}',
    cited_chunk_ids TEXT NOT NULL DEFAULT '[]',
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_document_versions_document_id
    ON document_versions(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document_version_id
    ON chunks(document_version_id);

CREATE INDEX IF NOT EXISTS idx_jobs_state
    ON jobs(state);
"""


def _connect(path: Path) -> sqlite3.Connection:
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_sqlite(settings: Settings) -> None:
    with sqlite_connection(settings) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        connection.executescript(SCHEMA_SQL)
        _upgrade_pdf_metadata_schema(connection)
        connection.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
            ("schema_version", "pdf-intelligence-v1"),
        )
        connection.commit()


def _upgrade_pdf_metadata_schema(connection: sqlite3.Connection) -> None:
    """Add PDF intelligence metadata to databases created by older releases."""
    additions = {
        "document_versions": {
            "layout_version": "TEXT",
            "document_type": "TEXT",
            "document_structure_tree": "TEXT NOT NULL DEFAULT '[]'",
        },
        "chunks": {
            "page_range": "TEXT",
            "bbox_list": "TEXT NOT NULL DEFAULT '[]'",
            "content_type": "TEXT NOT NULL DEFAULT 'text'",
            "source_type": "TEXT NOT NULL DEFAULT 'native_text'",
            "ocr_confidence": "REAL",
            "block_types": "TEXT NOT NULL DEFAULT '[]'",
            "table_markdown": "TEXT",
            "image_path": "TEXT",
            "caption": "TEXT",
            "image_metadata": "TEXT",
        },
    }
    for table, columns in additions.items():
        existing = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, definition in columns.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@contextmanager
def sqlite_connection(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = _connect(settings.database_path)
    try:
        connection.execute("PRAGMA foreign_keys=ON;")
        yield connection
    finally:
        connection.close()


def sqlite_health(settings: Settings) -> dict[str, bool | str]:
    with sqlite_connection(settings) as connection:
        one = connection.execute("SELECT 1").fetchone()[0]
        compile_options = {
            row[0] for row in connection.execute("PRAGMA compile_options").fetchall()
        }
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    return {
        "ok": one == 1,
        "journal_mode": journal_mode,
        "fts5_enabled": "ENABLE_FTS5" in compile_options,
    }
