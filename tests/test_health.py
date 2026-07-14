from pathlib import Path

from fastapi.testclient import TestClient

from enterprise_document_rag.config import Settings
from enterprise_document_rag.main import create_app


def test_live_health_returns_200(tmp_path: Path) -> None:
    settings = Settings(database_url=str(tmp_path / "agent.db"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_health_initializes_sqlite_wal(tmp_path: Path) -> None:
    settings = Settings(database_url=str(tmp_path / "agent.db"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["journal_mode"] == "wal"
    assert body["fts5_enabled"] is True

