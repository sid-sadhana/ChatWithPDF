"""RAGAS evaluation: faithfulness, relevancy, context precision."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class EvalService:
    def __init__(self, ollama_base_url: str, ollama_model: str) -> None:
        self._available = False
        self._base_url = ollama_base_url
        self._model = ollama_model
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.metrics import (
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
            )

            llm = LangchainLLMWrapper(
                ChatOllama(model=ollama_model, base_url=ollama_base_url)
            )
            embeddings = LangchainEmbeddingsWrapper(
                OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
            )

            self._metrics = [
                Faithfulness(llm=llm),
                ResponseRelevancy(llm=llm, embeddings=embeddings),
                LLMContextPrecisionWithoutReference(llm=llm),
            ]
            self._available = True
            log.info("RAGAS eval service ready (model=%s)", ollama_model)
        except Exception as exc:
            log.warning("RAGAS eval unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def evaluate(self, qa_pairs: list[dict[str, Any]]) -> dict:
        if not self._available:
            return {"error": "RAGAS not available", "scores": []}

        from ragas import SingleTurnSample, EvaluationDataset, evaluate

        samples = [
            SingleTurnSample(
                user_input=qa["question"],
                response=qa["answer"],
                retrieved_contexts=qa["contexts"],
            )
            for qa in qa_pairs
        ]

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=dataset, metrics=self._metrics)

        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            per_question.append({
                "question": row.get("user_input", ""),
                "answer": row.get("response", ""),
                "faithfulness": _safe_float(row.get("faithfulness")),
                "answer_relevancy": _safe_float(row.get("response_relevancy")),
                "context_precision": _safe_float(
                    row.get("llm_context_precision_without_reference")
                ),
            })

        aggregate = {}
        for key in ("faithfulness", "answer_relevancy", "context_precision"):
            vals = [q[key] for q in per_question if q[key] is not None]
            aggregate[key] = round(sum(vals) / len(vals), 4) if vals else None

        return {"aggregate": aggregate, "scores": per_question}


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return round(f, 4) if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
