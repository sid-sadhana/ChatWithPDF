"""HTTP API routes: upload PDF, chat, manage conversations."""
from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.services.exceptions import BadRequestError

if TYPE_CHECKING:
    from app.services.chat_service import ChatService

api_bp = Blueprint("api", __name__)


def _svc() -> ChatService:
    return current_app.extensions["chat_service"]


# ── Upload PDF → create conversation ─────────────────────────────
@api_bp.post("/upload")
def upload_pdf():
    file = request.files.get("file")
    if not file or not file.filename:
        raise BadRequestError("file is required")
    if not file.filename.lower().endswith(".pdf"):
        raise BadRequestError("only PDF files are supported")

    filename = secure_filename(file.filename)
    file_uuid = str(uuid.uuid4())
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_folder, f"{file_uuid}_{filename}")
    file.save(filepath)

    try:
        conversation = _svc().ingest_pdf(
            filepath=filepath,
            namespace=file_uuid,
            pdf_filename=filename,
        )
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass

    return jsonify(conversation), 201


# ── Chat within a conversation ────────────────────────────────────
@api_bp.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    conversation_id = (data.get("conversation_id") or "").strip()
    query = (data.get("query") or "").strip()

    if not conversation_id:
        raise BadRequestError("conversation_id is required")
    if not query:
        raise BadRequestError("query is required")

    result = _svc().chat(conversation_id=conversation_id, query=query)
    return jsonify(result), 200


# ── List conversations ────────────────────────────────────────────
@api_bp.get("/conversations")
def list_conversations():
    convos = _svc().conversations.list_all()
    return jsonify(convos), 200


# ── Get a single conversation with messages ───────────────────────
@api_bp.get("/conversations/<conversation_id>")
def get_conversation(conversation_id: str):
    conv = _svc().conversations.get(conversation_id)
    if not conv:
        return jsonify(error="Conversation not found"), 404
    return jsonify(conv), 200


# ── Delete a conversation ─────────────────────────────────────────
@api_bp.delete("/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    svc = _svc()
    conv = svc.conversations.get(conversation_id)
    if conv:
        svc.store.delete_namespace(conv["namespace"])
    deleted = svc.conversations.delete(conversation_id)
    if not deleted:
        return jsonify(error="Conversation not found"), 404
    return jsonify(ok=True), 200


# ── Delete all conversations + vector data ───────────────────────
@api_bp.delete("/conversations")
def delete_all():
    svc = _svc()
    for conv in svc.conversations.list_all():
        svc.store.delete_namespace(conv["namespace"])
        svc.conversations.delete(conv["conversation_id"])
    return jsonify(ok=True), 200


# ── RAGAS evaluation on a conversation ───────────────────────────
@api_bp.post("/conversations/<conversation_id>/evaluate")
def evaluate_conversation(conversation_id: str):
    svc = _svc()

    if not svc.eval_service.available:
        return jsonify(error="RAGAS evaluation is not available (missing dependencies)"), 503

    conv = svc.conversations.get(conversation_id)
    if not conv:
        return jsonify(error="Conversation not found"), 404

    messages = conv.get("messages", [])
    qa_pairs = []
    for i, msg in enumerate(messages):
        if (
            msg["role"] == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ):
            search_results = svc.store.search(
                query_text=msg["content"],
                namespace=conv["namespace"],
                top_k=svc.retrieval_top_k,
            )
            contexts = [r["text"] for r in search_results["fused"]]
            qa_pairs.append({
                "question": msg["content"],
                "answer": messages[i + 1]["content"],
                "contexts": contexts,
            })

    if not qa_pairs:
        raise BadRequestError("No Q&A pairs found in conversation")

    result = svc.eval_service.evaluate(qa_pairs)
    return jsonify(result), 200
