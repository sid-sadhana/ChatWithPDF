"""Ollama LLM wrapper."""
from __future__ import annotations

import logging

from ollama import Client

from app.services.exceptions import UpstreamError

log = logging.getLogger(__name__)


class LLMService:
    def __init__(self, base_url: str, model: str) -> None:
        self._client = Client(host=base_url)
        self._model = model
        log.info("Ollama LLM: %s model=%s", base_url, model)

    def generate(self, system: str, user: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        try:
            resp = self._client.chat(
                model=self._model,
                messages=messages,
                stream=False,
                options={"temperature": 0.3},
                think=False,
            )
            return resp.message.content or ""
        except Exception as exc:
            raise UpstreamError(f"Ollama call failed: {exc}") from exc
