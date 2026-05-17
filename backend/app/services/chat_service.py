"""Top-level service: naive RAG with Pinecone + Ollama."""
from __future__ import annotations

import logging
from typing import Any, Mapping

from app.services.conversation_service import ConversationService
from app.services.eval_service import EvalService
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.reranker_service import RerankerService
from app.services.vector_store import VectorStore

log = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided PDF context.
Use ONLY the context below to answer. If the context doesn't contain enough information, say so.
Be concise and accurate. Use plain text only. Never use markdown, asterisks, bullet points, or any formatting symbols.

Context:
{context}"""


class ChatService:
    def __init__(
        self,
        pdf: PDFService,
        store: VectorStore,
        llm: LLMService,
        reranker: RerankerService,
        eval_service: EvalService,
        conversations: ConversationService,
        retrieval_top_k: int,
    ) -> None:
        self.pdf = pdf
        self.store = store
        self.llm = llm
        self.reranker = reranker
        self.eval_service = eval_service
        self.conversations = conversations
        self.retrieval_top_k = retrieval_top_k

    def ingest_pdf(self, filepath: str, namespace: str, pdf_filename: str) -> dict:
        content = self.pdf.extract(filepath)
        self.store.upsert_texts(content.text_chunks, namespace)
        log.info("Indexed %d text chunks into namespace %s", len(content.text_chunks), namespace)

        conversation = self.conversations.create(
            pdf_filename=pdf_filename, namespace=namespace
        )
        return conversation

    def chat(self, conversation_id: str, query: str) -> dict:
        conv = self.conversations.get(conversation_id)
        if not conv:
            from app.services.exceptions import NotFoundError
            raise NotFoundError(f"Conversation {conversation_id} not found")

        self.conversations.add_message(conversation_id, role="user", content=query)

        search_results = self.store.search(
            query_text=query,
            namespace=conv["namespace"],
            top_k=self.retrieval_top_k * 4,
        )

        dense_hits = search_results["dense"]
        sparse_hits = search_results["sparse"]
        fused_hits = search_results["fused"]
        reranked_hits = self.reranker.rerank(query, fused_hits)

        context_chunks = reranked_hits if reranked_hits else fused_hits
        context = "\n\n---\n\n".join(r["text"] for r in context_chunks) if context_chunks else "No relevant content found."

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in conv.get("messages", [])[-10:]
        ]

        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        answer = self.llm.generate(system=system_prompt, user=query, history=history)

        msg = self.conversations.add_message(
            conversation_id, role="assistant", content=answer
        )

        top_k = self.retrieval_top_k
        return {
            "message": msg,
            "conversation_id": conversation_id,
            "sources": {
                "dense": dense_hits[:top_k],
                "sparse": sparse_hits[:top_k],
                "fused": fused_hits[:top_k],
                "reranked": reranked_hits[:top_k],
            },
        }


def build_chat_service(config: Mapping[str, Any]) -> ChatService:
    pdf = PDFService(
        chunk_size=int(config["CHUNK_SIZE"]),
        chunk_overlap=int(config["CHUNK_OVERLAP"]),
    )
    store = VectorStore(
        api_key=str(config["PINECONE_API_KEY"]),
        dense_host=str(config["PINECONE_DENSE_HOST"]),
        sparse_host=str(config["PINECONE_SPARSE_HOST"]),
        alpha=float(config["HYBRID_ALPHA"]),
    )
    llm = LLMService(
        base_url=str(config["OLLAMA_BASE_URL"]),
        model=str(config["OLLAMA_MODEL"]),
    )
    reranker = RerankerService(
        model_name=str(config["RERANKER_MODEL"]),
        top_k=int(config["RETRIEVAL_TOP_K"]),
    )
    eval_svc = EvalService(
        ollama_base_url=str(config["OLLAMA_BASE_URL"]),
        ollama_model=str(config["OLLAMA_MODEL"]),
    )
    conversations = ConversationService()

    return ChatService(
        pdf=pdf,
        store=store,
        llm=llm,
        reranker=reranker,
        eval_service=eval_svc,
        conversations=conversations,
        retrieval_top_k=int(config["RETRIEVAL_TOP_K"]),
    )
