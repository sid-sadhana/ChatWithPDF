"""Cross-encoder reranker using Qwen3-VL-Reranker-2B."""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.exceptions import UpstreamError

log = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str, top_k: int) -> None:
        self._top_k = top_k
        self._available = False
        try:
            log.info("Loading reranker: %s", model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32
            )
            self._model.eval()
            self._available = True
            log.info("Reranker loaded successfully")
        except Exception as exc:
            log.warning("Reranker failed to load, will skip reranking: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        if not self._available or not results:
            return results[: self._top_k]

        pairs = [[query, r["text"]] for r in results]

        try:
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                scores = self._model(**inputs).logits.squeeze(-1).tolist()
        except Exception as exc:
            log.warning("Reranking failed, returning RRF order: %s", exc)
            return results[: self._top_k]

        if isinstance(scores, float):
            scores = [scores]

        for r, s in zip(results, scores):
            r["rerank_score"] = round(s, 4)

        reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return reranked[: self._top_k]
