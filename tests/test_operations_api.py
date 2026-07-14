from pathlib import Path
from time import monotonic, sleep
from zipfile import ZipFile

from fastapi.testclient import TestClient

from enterprise_document_rag.config import Settings
from enterprise_document_rag.db import initialize_sqlite, sqlite_connection
from enterprise_document_rag.fingerprinting import streaming_sha256
from enterprise_document_rag.main import create_app
from enterprise_document_rag.repositories import (
    DocumentRepository,
    JobRepository,
    KnowledgeBaseRepository,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=str(tmp_path / "agent.db"),
        qdrant_path=tmp_path / "qdrant",
        embedding_backend="hash",
        vector_collection_name="document_chunks",
        llm_backend="extractive",
        authorized_roots=str(tmp_path),
    )


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        payload = client.get(f"/api/v1/jobs/{job_id}").json()
        if payload["state"] in {"succeeded", "failed"}:
            return payload
        sleep(0.05)
    raise AssertionError(f"job did not finish: {job_id}")


def test_non_developer_flow_can_add_source_index_and_query(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "washer.md").write_text(
        "# Washer Parameters\n\nCleaning pressure is 0.25 MPa.",
        encoding="utf-8",
    )
    settings = _settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "washer"})
        assert knowledge_base.status_code == 200
        knowledge_base_id = knowledge_base.json()["id"]

        created_source = client.post(
            "/api/v1/sources",
            json={"knowledge_base_id": knowledge_base_id, "root_path": str(source)},
        )
        assert created_source.status_code == 200

        scan = client.post(f"/api/v1/sources/{created_source.json()['id']}/scan")
        assert scan.status_code == 200
        assert scan.json()["counts"]["add"] == 1
        job = _wait_for_job(client, scan.json()["jobs"][0]["id"])
        assert job["state"] == "succeeded"

        documents = client.get("/api/v1/documents", params={"knowledge_base_id": knowledge_base_id})
        assert documents.status_code == 200
        document = documents.json()[0]
        assert document["version_state"] == "ready"
        assert document["chunk_count"] == 1

        answer = client.post(
            "/api/v1/query",
            json={"knowledge_base_id": knowledge_base_id, "question": "cleaning pressure"},
        )
        assert answer.status_code == 200
        payload = answer.json()
        assert payload["insufficient_evidence"] is False
        assert payload["citations"][0]["file_name"] == "washer.md"
        assert "0.25 MPa" in payload["citations"][0]["quote"]

        source_file = client.get(f"/api/v1/documents/{document['document_id']}/source-file")
        assert source_file.status_code == 200

        preview = client.get(
            f"/api/v1/documents/{document['document_id']}/preview",
            params={"chunk_id": answer.json()["citations"][0]["chunk_id"]},
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("text/html")
        assert "Cleaning pressure is 0.25 MPa." in preview.text

        field_search = client.post(
            "/api/v1/field-search",
            json={"knowledge_base_id": knowledge_base_id, "fields": ["Cleaning pressure"]},
        )
        assert field_search.status_code == 200
        match = field_search.json()[0]["files"][0]
        assert match["file_name"] == "washer.md"
        assert match["canonical_path"] == str(source.resolve() / "washer.md")

        reindex = client.post(f"/api/v1/documents/{document['document_id']}/reindex")
        assert reindex.status_code == 200
        assert _wait_for_job(client, reindex.json()["id"])["state"] == "succeeded"


def test_application_startup_resumes_queued_job(tmp_path: Path) -> None:
    source = tmp_path / "resume.md"
    source.write_text("# Scope\n\nThe supported flow rate is 20 L/min.", encoding="utf-8")
    settings = _settings(tmp_path)
    initialize_sqlite(settings)
    with sqlite_connection(settings) as connection:
        knowledge_base = KnowledgeBaseRepository(connection).create(
            name="resume-kb",
            embedding_model="hash-cpu",
            vector_backend="qdrant_local",
        )
        documents = DocumentRepository(connection)
        document = documents.create(
            knowledge_base_id=knowledge_base.id,
            canonical_path=source.resolve(),
        )
        fingerprint = streaming_sha256(source)
        documents.create_version(
            document_id=document.id,
            sha256=fingerprint,
            size_bytes=source.stat().st_size,
            mtime_ns=source.stat().st_mtime_ns,
            parser_version="parser-v1",
        )
        job = JobRepository(connection).enqueue(
            knowledge_base_id=knowledge_base.id,
            operation="add",
            path=source.resolve(),
            expected_sha256=fingerprint,
        )
        assert job.state == "queued"

    with TestClient(create_app(settings)) as client:
        resumed = _wait_for_job(client, job.id)
        assert resumed["state"] == "succeeded"
        documents = client.get("/api/v1/documents", params={"knowledge_base_id": knowledge_base.id})
        assert documents.json()[0]["version_state"] == "ready"


def test_delete_knowledge_base_removes_documents_and_vectors(tmp_path: Path) -> None:
    source = tmp_path / "delete-me.md"
    source.write_text("# Retention\n\nDelete this indexed content.", encoding="utf-8")
    settings = _settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "delete-me"})
        knowledge_base_id = knowledge_base.json()["id"]
        created_source = client.post(
            "/api/v1/sources",
            json={"knowledge_base_id": knowledge_base_id, "root_path": str(tmp_path)},
        )
        scan = client.post(f"/api/v1/sources/{created_source.json()['id']}/scan")
        _wait_for_job(client, scan.json()["jobs"][0]["id"])

        deleted = client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted_vectors"] == 1
        assert client.get("/api/v1/knowledge-bases").json() == []
        assert (
            client.get("/api/v1/sources", params={"knowledge_base_id": knowledge_base_id}).json()
            == []
        )
        assert (
            client.get("/api/v1/documents", params={"knowledge_base_id": knowledge_base_id}).json()
            == []
        )


def test_knowledge_base_can_authorize_multiple_source_directories(tmp_path: Path) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()
    settings = _settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "multi-source"})
        knowledge_base_id = knowledge_base.json()["id"]
        for source in (first_source, second_source):
            response = client.post(
                "/api/v1/sources",
                json={"knowledge_base_id": knowledge_base_id, "root_path": str(source)},
            )
            assert response.status_code == 200

        sources = client.get("/api/v1/sources", params={"knowledge_base_id": knowledge_base_id})
        assert {item["root_path"] for item in sources.json()} == {
            str(first_source.resolve()),
            str(second_source.resolve()),
        }


def test_failed_documents_endpoint_returns_paths_and_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    failed_path = tmp_path / "unreadable.docx"

    with TestClient(create_app(settings)) as client:
        knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "failed-docs"})
        knowledge_base_id = knowledge_base.json()["id"]
        with sqlite_connection(settings) as connection:
            document = DocumentRepository(connection).create(
                knowledge_base_id=knowledge_base_id,
                canonical_path=failed_path,
            )
            version = DocumentRepository(connection).create_version(
                document_id=document.id,
                sha256="f" * 64,
                size_bytes=0,
                mtime_ns=0,
                parser_version="parser-v1",
                state="failed",
            )
            connection.execute(
                "UPDATE document_versions SET error = ? WHERE id = ?",
                ("DOCX package is invalid", version.id),
            )
            connection.commit()

        failed = client.get(
            "/api/v1/documents/failed", params={"knowledge_base_id": knowledge_base_id}
        )
        assert failed.status_code == 200
        result = failed.json()
        assert len(result) == 1
        assert result[0]["document_id"] == document.id
        assert result[0]["canonical_path"] == str(failed_path)
        assert result[0]["error"] == "DOCX package is invalid"


def test_zip_source_is_scanned_indexed_and_searchable_by_internal_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    archive_path = source / "tender-documents.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("specifications/pressure.txt", "Archived pressure is 0.25 MPa.")
    settings = _settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        knowledge_base = client.post("/api/v1/knowledge-bases", json={"name": "zip-source"})
        knowledge_base_id = knowledge_base.json()["id"]
        created_source = client.post(
            "/api/v1/sources",
            json={"knowledge_base_id": knowledge_base_id, "root_path": str(source)},
        )
        scan = client.post(f"/api/v1/sources/{created_source.json()['id']}/scan")
        job = _wait_for_job(client, scan.json()["jobs"][0]["id"])
        assert job["state"] == "succeeded"

        matches = client.post(
            "/api/v1/field-search",
            json={"knowledge_base_id": knowledge_base_id, "fields": ["Archived pressure"]},
        )
        match = matches.json()[0]["files"][0]
        assert match["file_name"] == "tender-documents.zip"
        assert "压缩包内文件：specifications/pressure.txt" in match["quote"]
