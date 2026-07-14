import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    score: float
    payload: dict[str, str | int | float | bool | None]


class VectorStore(Protocol):
    def upsert(self, *, collection_name: str, records: list[VectorRecord], dimension: int) -> None:
        ...

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        filter_conditions: dict[str, str | list[str]] | None = None,
    ) -> list[VectorSearchResult]: ...

    def count(self, *, collection_name: str) -> int: ...

    def delete_by_filter(
        self,
        *,
        filter_conditions: dict[str, str | list[str]],
    ) -> int: ...


class QdrantLocalVectorStore:
    def __init__(self, *, path: str = ":memory:") -> None:
        self.client = QdrantClient(path=path)
        self._lock = threading.RLock()

    def upsert(self, *, collection_name: str, records: list[VectorRecord], dimension: int) -> None:
        with self._lock:
            self._ensure_collection(collection_name=collection_name, dimension=dimension)
            if not records:
                return
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload,
                    )
                    for record in records
                ],
            )

    def count(self, *, collection_name: str) -> int:
        with self._lock:
            if not self.client.collection_exists(collection_name):
                return 0
            return int(self.client.count(collection_name=collection_name, exact=True).count)

    def delete_by_filter(
        self,
        *,
        filter_conditions: dict[str, str | list[str]],
    ) -> int:
        """Delete matching points from every local collection.

        Collections can change when an embedding model is migrated. Clearing all
        matching payloads keeps a deleted knowledge base out of both current and
        historical embedding collections.
        """
        query_filter = self._build_filter(filter_conditions)
        if query_filter is None:
            return 0
        with self._lock:
            deleted = 0
            for collection in self.client.get_collections().collections:
                collection_name = collection.name
                deleted += int(
                    self.client.count(
                        collection_name=collection_name,
                        count_filter=query_filter,
                        exact=True,
                    ).count
                )
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=query_filter),
                    wait=True,
                )
            return deleted

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        filter_conditions: dict[str, str | list[str]] | None = None,
    ) -> list[VectorSearchResult]:
        with self._lock:
            if not self.client.collection_exists(collection_name):
                return []
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=self._build_filter(filter_conditions or {}),
                limit=limit,
                with_payload=True,
            )
            return [
                VectorSearchResult(
                    id=str(point.id),
                    score=float(point.score),
                    payload=dict(point.payload or {}),
                )
                for point in response.points
            ]

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Block vector reads and writes while related metadata is changed."""
        with self._lock:
            yield

    def _ensure_collection(self, *, collection_name: str, dimension: int) -> None:
        if self.client.collection_exists(collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def _build_filter(
        self,
        filter_conditions: dict[str, str | list[str]],
    ) -> models.Filter | None:
        if not filter_conditions:
            return None
        must: list[models.FieldCondition] = []
        for key, value in filter_conditions.items():
            if isinstance(value, list):
                must.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                must.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=must)


_LOCAL_STORES: dict[str, QdrantLocalVectorStore] = {}
_LOCAL_STORES_LOCK = threading.Lock()


def get_local_vector_store(*, path: str) -> QdrantLocalVectorStore:
    """Return the process-wide client for a local Qdrant storage directory."""
    key = path if path == ":memory:" else str(Path(path).resolve())
    with _LOCAL_STORES_LOCK:
        if key not in _LOCAL_STORES:
            _LOCAL_STORES[key] = QdrantLocalVectorStore(path=path)
        return _LOCAL_STORES[key]


def close_local_vector_store(*, path: str) -> None:
    key = path if path == ":memory:" else str(Path(path).resolve())
    with _LOCAL_STORES_LOCK:
        store = _LOCAL_STORES.pop(key, None)
    if store is not None:
        store.client.close()
