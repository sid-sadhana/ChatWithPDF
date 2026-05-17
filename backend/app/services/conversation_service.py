"""In-memory conversation storage."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


class ConversationService:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        log.info("Using in-memory conversation store")

    def create(self, pdf_filename: str, namespace: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        cid = str(uuid.uuid4())
        doc = {
            "conversation_id": cid,
            "title": pdf_filename,
            "pdf_filename": pdf_filename,
            "namespace": namespace,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._store[cid] = doc
        return dict(doc)

    def get(self, conversation_id: str) -> dict | None:
        doc = self._store.get(conversation_id)
        return dict(doc) if doc else None

    def list_all(self, limit: int = 50) -> list[dict]:
        convs = sorted(
            self._store.values(), key=lambda d: d["updated_at"], reverse=True
        )[:limit]
        return [{k: v for k, v in c.items() if k != "messages"} for c in convs]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        msg = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        doc = self._store.get(conversation_id)
        if doc:
            doc["messages"].append(msg)
            doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        return msg

    def delete(self, conversation_id: str) -> bool:
        return self._store.pop(conversation_id, None) is not None
