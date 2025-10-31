from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - explicit guidance for missing dependency
    raise RuntimeError(
        "chromadb is required for persistent memory. Install dependencies via "
        "`pip install -r requirements.txt`."
    ) from exc

DEFAULT_DB_PATH = Path(os.environ.get("AGENT_MEMORY_DB_PATH", "./agent_memory_db"))
DEFAULT_COLLECTION_NAME = os.environ.get("AGENT_MEMORY_COLLECTION", "research_and_development")


@dataclass
class MemoryRecord:
    """Lightweight container representing a document stored in ChromaDB."""

    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the record."""
        payload: Dict[str, Any] = {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
        }
        if self.distance is not None:
            payload["distance"] = self.distance
        return payload


class ChromaMemory:
    """
    Thin wrapper around a persistent ChromaDB collection.

    Provides convenience helpers for upserts, similarity queries, and basic CRUD
    while keeping the underlying collection accessible to other modules.
    """

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any = None,
    ) -> None:
        db_path = Path(path or DEFAULT_DB_PATH).expanduser().resolve()
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    @property
    def client(self):
        """Expose the underlying Chroma client."""
        return self._client

    @property
    def collection(self):
        """Expose the underlying Chroma collection."""
        return self._collection

    def upsert(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert or update a document in the collection."""
        payload_metadata = metadata or {}
        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[payload_metadata],
        )

    def get(self, doc_id: str) -> Optional[MemoryRecord]:
        """Return a single record by ID, or None if the document is missing."""
        response = self._collection.get(ids=[doc_id])
        ids: List[str] = response.get("ids") or []
        if not ids:
            return None
        documents: List[str] = response.get("documents") or [""]
        metadatas: List[Dict[str, Any]] = response.get("metadatas") or [{}]
        return MemoryRecord(
            id=ids[0],
            document=documents[0],
            metadata=metadatas[0] or {},
        )

    def delete(self, doc_ids: Iterable[str]) -> None:
        """Delete one or more documents from the collection."""
        ids = list(doc_ids)
        if not ids:
            return
        self._collection.delete(ids=ids)

    def clear(self) -> None:
        """Remove every record in the collection."""
        self._collection.delete(where={})

    def iter_all(self) -> List[MemoryRecord]:
        """Return all stored records."""
        response = self._collection.get()
        ids: List[str] = response.get("ids") or []
        documents: List[str] = response.get("documents") or []
        metadatas: List[Dict[str, Any]] = response.get("metadatas") or []

        records: List[MemoryRecord] = []
        for idx, doc, meta in zip_longest(ids, documents, metadatas, fillvalue=None):
            if idx is None or doc is None:
                continue
            records.append(
                MemoryRecord(
                    id=idx,
                    document=doc,
                    metadata=meta or {},
                )
            )
        return records

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Sequence[str]] = None,
    ) -> List[MemoryRecord]:
        """
        Run a similarity search against the collection using the supplied text.

        Returns a list of MemoryRecord objects sorted by relevance.
        """
        if not query_text.strip():
            return []

        response = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=include,
        )

        ids = list((response.get("ids") or [[]])[0])
        documents = list((response.get("documents") or [[]])[0])
        metadatas = list((response.get("metadatas") or [[]])[0])
        distances = list((response.get("distances") or [[]])[0])

        if not distances:
            distances = [None] * len(ids)
        elif len(distances) < len(ids):
            distances.extend([None] * (len(ids) - len(distances)))

        records: List[MemoryRecord] = []
        for idx, doc, meta, dist in zip_longest(ids, documents, metadatas, distances, fillvalue=None):
            if idx is None or doc is None:
                continue
            records.append(
                MemoryRecord(
                    id=idx,
                    document=doc,
                    metadata=meta or {},
                    distance=dist,
                )
            )
        return records


_shared_memory = ChromaMemory()

# Backwards-compatible exports used by the rest of the codebase.
client = _shared_memory.client
collection = _shared_memory.collection


def store_memory(doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Module-level helper retained for compatibility with earlier imports."""
    _shared_memory.upsert(doc_id=doc_id, document=document, metadata=metadata or {})


def retrieve_memories(query_text: str, n_results: int = 2) -> List[str]:
    """Return the raw documents for the best matching memories."""
    records = _shared_memory.query(query_text=query_text, n_results=n_results)
    return [record.document for record in records]
