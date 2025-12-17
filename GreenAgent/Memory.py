from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULT_DB_PATH = Path(os.environ.get("AGENT_MEMORY_DB_PATH", "agent_memory.json"))
DEFAULT_COLLECTION_NAME = os.environ.get("AGENT_MEMORY_COLLECTION", "research_and_development")

@dataclass
class MemoryRecord:
    """Lightweight container representing a document."""
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document": self.document,
            "metadata": self.metadata,
        }

class SimpleMemory:
    """
    Simple file-based memory replacement for ChromaDB.
    Stores records in a local JSON file.
    """

    def __init__(
        self,
        path: Optional[os.PathLike[str] | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any = None, # Ignored, kept for compat
    ) -> None:
        self.path = Path(path or DEFAULT_DB_PATH).resolve()
        
        # We ignore collection_name for single-file simple memory, 
        # or we could use it to key the JSON. Let's keep it simple: flat list/dict.
        self.data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                content = self.path.read_text()
                if content.strip():
                    self.data = json.loads(content)
                else:
                    self.data = {}
            except Exception as e:
                print(f"Warning: Failed to load memory file {self.path}: {e}")
                self.data = {}
        else:
            self.data = {}

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.data, indent=2))
        except Exception as e:
            print(f"Error: Failed to save memory to {self.path}: {e}")

    @property
    def client(self):
        return self

    @property
    def collection(self):
        return self

    def upsert(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert or update a document."""
        self.data[doc_id] = {
            "id": doc_id,
            "document": document,
            "metadata": metadata or {},
        }
        self._save()

    def get(self, doc_id: str) -> Optional[MemoryRecord]:
        """Return a single record by ID."""
        record_dict = self.data.get(doc_id)
        if not record_dict:
            return None
        return MemoryRecord(
            id=record_dict["id"],
            document=record_dict["document"],
            metadata=record_dict["metadata"],
        )

    def delete(self, doc_ids: Iterable[str]) -> None:
        """Delete one or more documents."""
        changed = False
        for doc_id in doc_ids:
            if doc_id in self.data:
                del self.data[doc_id]
                changed = True
        if changed:
            self._save()

    def clear(self) -> None:
        """Remove every record."""
        self.data = {}
        self._save()

    def iter_all(self) -> List[MemoryRecord]:
        """Return all stored records."""
        return [
            MemoryRecord(
                id=r["id"],
                document=r["document"],
                metadata=r["metadata"]
            )
            for r in self.data.values()
        ]

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[Sequence[str]] = None,
    ) -> List[MemoryRecord]:
        """
        Naive query: returns the most recently added records.
        Ignores semantic similarity since we have no embeddings.
        """
        # Convert to list and take last N (assuming insertion order is preserved in dicts >= 3.7)
        all_records = list(self.data.values())
        # Reverse to get most recent first
        recent_records = all_records[::-1][:n_results]
        
        return [
            MemoryRecord(
                id=r["id"],
                document=r["document"],
                metadata=r["metadata"],
                distance=0.0 # Dummy distance
            )
            for r in recent_records
        ]


_shared_memory = SimpleMemory()

# Backwards-compatible exports
ChromaMemory = SimpleMemory
client = _shared_memory.client
collection = _shared_memory.collection


def store_memory(doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    _shared_memory.upsert(doc_id=doc_id, document=document, metadata=metadata or {})


def retrieve_memories(query_text: str, n_results: int = 2) -> List[str]:
    records = _shared_memory.query(query_text=query_text, n_results=n_results)
    return [record.document for record in records]
