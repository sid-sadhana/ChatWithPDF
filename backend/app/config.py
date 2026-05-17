"""Configuration classes for the Flask app."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

    # --- Pinecone ---
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_DENSE_HOST = os.getenv("PINECONE_DENSE_HOST", "")
    PINECONE_SPARSE_HOST = os.getenv("PINECONE_SPARSE_HOST", "")
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))

    # --- Ollama ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    # --- Reranker ---
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")

    # --- RAG ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # --- Uploads ---
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(INSTANCE_DIR / "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "25")) * 1024 * 1024

    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    def __init__(self) -> None:
        if not self.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY must be set in production.")


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    PINECONE_API_KEY = "test"


_CONFIGS: dict[str, type[Config]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name: str | None = None) -> type[Config]:
    name = (name or os.getenv("FLASK_ENV", "development")).lower()
    return _CONFIGS.get(name, DevelopmentConfig)
