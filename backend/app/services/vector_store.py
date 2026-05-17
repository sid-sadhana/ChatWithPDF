"""Hybrid search: dense (llama-text-embed-v2) + sparse (pinecone-sparse-english-v0)."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pinecone import Pinecone

from app.services.exceptions import UpstreamError

log = logging.getLogger(__name__)

RRF_K = 60


def _reciprocal_rank_fusion(
    dense_hits: list[dict], sparse_hits: list[dict], alpha: float
) -> list[dict]:
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    source: dict[str, str] = {}

    for rank, h in enumerate(dense_hits):
        rid = h["id"]
        scores[rid] = scores.get(rid, 0) + alpha * (1.0 / (RRF_K + rank + 1))
        texts[rid] = h["text"]
        source[rid] = "dense"

    for rank, h in enumerate(sparse_hits):
        rid = h["id"]
        scores[rid] = scores.get(rid, 0) + (1 - alpha) * (1.0 / (RRF_K + rank + 1))
        if rid not in texts:
            texts[rid] = h["text"]
            source[rid] = "sparse"
        else:
            source[rid] = "both"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"text": texts[rid], "score": round(score, 6), "match": source[rid]}
        for rid, score in ranked
        if texts.get(rid)
    ]


class VectorStore:
    def __init__(
        self,
        api_key: str,
        dense_host: str,
        sparse_host: str,
        alpha: float = 0.7,
    ) -> None:
        self._client = Pinecone(api_key=api_key)
        self._dense = self._client.Index(host=dense_host)
        self._sparse = self._client.Index(host=sparse_host)
        self._alpha = alpha
        log.info("Hybrid search: dense=%s sparse=%s alpha=%.1f", dense_host, sparse_host, alpha)

    def upsert_texts(self, texts: list[str], namespace: str) -> None:
        records = [
            {"_id": str(uuid.uuid4()), "text": text}
            for text in texts
        ]
        batch_size = 50
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                self._dense.upsert_records(namespace, batch)
                self._sparse.upsert_records(namespace, batch)
        except Exception as exc:
            raise UpstreamError(f"Pinecone upsert failed: {exc}") from exc
        log.info("Upserted %d records to dense+sparse, namespace %s", len(records), namespace)

    def search(self, query_text: str, namespace: str, top_k: int) -> dict:
        fetch_k = top_k * 3

        dense_hits = self._search_index(self._dense, query_text, namespace, fetch_k)
        sparse_hits = self._search_index(self._sparse, query_text, namespace, fetch_k)

        fused = _reciprocal_rank_fusion(dense_hits, sparse_hits, self._alpha)

        return {
            "dense": dense_hits[:top_k],
            "sparse": sparse_hits[:top_k],
            "fused": fused[:top_k],
        }

    def _search_index(
        self, index: Any, query_text: str, namespace: str, top_k: int
    ) -> list[dict]:
        try:
            res = index.search(
                namespace=namespace,
                query={"inputs": {"text": query_text}, "top_k": top_k},
            )
        except Exception as exc:
            log.warning("Search failed on one index: %s", exc)
            return []

        hits = res.get("result", {}).get("hits", [])
        return [
            {"id": h.get("_id", ""), "text": h.get("fields", {}).get("text", ""), "score": h.get("_score", 0)}
            for h in hits
            if h.get("fields", {}).get("text")
        ]

    def delete_namespace(self, namespace: str) -> None:
        for idx in (self._dense, self._sparse):
            try:
                idx.delete(delete_all=True, namespace=namespace)
            except Exception as exc:
                log.warning("Pinecone delete failed: %s", exc)
