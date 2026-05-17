"""Pinecone vector store with integrated embedding (server-side)."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pinecone import Pinecone

from app.services.exceptions import UpstreamError

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, api_key: str, index_host: str) -> None:
        self._client = Pinecone(api_key=api_key)
        self._index = self._client.Index(host=index_host)
        log.info("Connected to Pinecone index: %s", index_host)

    def upsert_texts(self, texts: list[str], namespace: str) -> None:
        records = [
            {"_id": str(uuid.uuid4()), "text": text}
            for text in texts
        ]
        batch_size = 50
        try:
            for i in range(0, len(records), batch_size):
                self._index.upsert_records(namespace, records[i : i + batch_size])
        except Exception as exc:
            raise UpstreamError(f"Pinecone upsert failed: {exc}") from exc
        log.info("Upserted %d records into namespace %s", len(records), namespace)

    def search(self, query_text: str, namespace: str, top_k: int) -> list[dict]:
        try:
            res: Any = self._index.search(
                namespace=namespace,
                query={"inputs": {"text": query_text}, "top_k": top_k},
            )
        except Exception as exc:
            raise UpstreamError(f"Pinecone search failed: {exc}") from exc

        hits = res.get("result", {}).get("hits", [])
        return [
            {
                "text": h.get("fields", {}).get("text", ""),
                "score": h.get("_score", 0),
            }
            for h in hits
            if h.get("fields", {}).get("text")
        ]

    def delete_namespace(self, namespace: str) -> None:
        try:
            self._index.delete(delete_all=True, namespace=namespace)
        except Exception as exc:
            log.warning("Pinecone delete failed for %s: %s", namespace, exc)
