"""Persistent vector store for MemoryNote semantic search.

Separate from InteractionMemory — dedicated to note retrieval.
One ChromaDB collection per reasoning session (keyed by session_id).
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, List, Optional

import chromadb

from wemg.reasoning.interaction_memory import LocalCompatibleEmbedding, ThreadSafeReadWriteLock

if TYPE_CHECKING:
    from wemg.reasoning.working_memory import MemoryNote, NoteType

logger = logging.getLogger(__name__)


class NoteVectorStore:
    """Persistent ChromaDB-backed vector store for MemoryNote semantic retrieval.

    Thread-safe for concurrent reads; exclusive writes use a write lock.
    One ChromaDB collection per session to avoid cross-session contamination.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_base_url: str,
        embedding_model_name: str,
        embedding_api_key: str = "EMPTY",
        session_id: Optional[str] = None,
        cleanup_collection_on_close: bool = True,
    ):
        self._lock = ThreadSafeReadWriteLock()
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._cleanup_collection_on_close = cleanup_collection_on_close

        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = LocalCompatibleEmbedding(
            base_url=embedding_base_url,
            model_name=embedding_model_name,
            api_key=embedding_api_key,
        )
        self._collection_name = f"notes_{self._session_id}"
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
        )
        # In-memory note cache: avoids redundant ChromaDB deserialisation
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_or_update(self, note: "MemoryNote") -> None:
        """Upsert a note into the vector store."""
        meta = {
            "note_type": note.note_type.value,
            "provenance": note.provenance.value,
            "confidence": note.confidence,
            "hop_depth": note.hop_depth if note.hop_depth is not None else -1,
            "created_at_step": note.created_at_step,
            "note_json": note.to_json(),
        }
        with self._lock.write_lock():
            try:
                self._collection.upsert(
                    ids=[note.id],
                    documents=[note.content],
                    metadatas=[meta],
                )
                self._cache[note.id] = note
            except Exception as exc:
                logger.warning("NoteVectorStore upsert failed for note %s: %s", note.id, exc)

    def delete(self, note_id: str) -> None:
        """Remove a note from the vector store."""
        with self._lock.write_lock():
            try:
                self._collection.delete(ids=[note_id])
                self._cache.pop(note_id, None)
            except Exception as exc:
                logger.warning("NoteVectorStore delete failed for note %s: %s", note_id, exc)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        filter_types: Optional[List["NoteType"]] = None,
    ) -> List["MemoryNote"]:
        """Return up to *k* notes most semantically similar to *query*."""
        where = None
        if filter_types:
            type_values = [t.value if hasattr(t, "value") else str(t) for t in filter_types]
            if len(type_values) == 1:
                where = {"note_type": type_values[0]}
            else:
                where = {"note_type": {"$in": type_values}}

        with self._lock.read_lock():
            try:
                count = self._collection.count()
                if count == 0:
                    return []
                n_results = min(k, count)
                results = self._collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                )
                notes: List["MemoryNote"] = []
                for meta in (results.get("metadatas") or [[]])[0]:
                    note_json = meta.get("note_json")
                    if note_json:
                        from wemg.reasoning.working_memory import MemoryNote
                        note = MemoryNote.from_json(note_json)
                        notes.append(self._cache.get(note.id, note))
                return notes
            except Exception as exc:
                logger.warning("NoteVectorStore search failed: %s", exc)
                return []

    def get(self, note_id: str) -> Optional["MemoryNote"]:
        """Retrieve a single note by ID."""
        if note_id in self._cache:
            return self._cache[note_id]
        with self._lock.read_lock():
            try:
                result = self._collection.get(ids=[note_id], include=["metadatas"])
                metas = result.get("metadatas") or []
                if metas:
                    note_json = metas[0].get("note_json")
                    if note_json:
                        from wemg.reasoning.working_memory import MemoryNote
                        return MemoryNote.from_json(note_json)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._cleanup_collection_on_close:
            try:
                self._chroma_client.delete_collection(name=self._collection_name)
            except Exception as exc:
                logger.warning(
                    "NoteVectorStore collection cleanup failed for %s: %s",
                    self._collection_name,
                    exc,
                )
        try:
            self._embedding_fn.close()
        except Exception:
            pass


__all__ = ["NoteVectorStore"]
