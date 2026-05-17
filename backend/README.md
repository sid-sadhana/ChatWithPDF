# Backend -- ChatWithPDF API

Flask API with hybrid RAG pipeline: semantic chunking, dense + sparse search with RRF, cross-encoder reranking, RAGAS evaluation, and local LLM generation.

## Layout

```
backend/
├── app/
│   ├── __init__.py              # app factory
│   ├── config.py                # env-driven config
│   ├── api/
│   │   ├── routes.py            # /api/upload, /api/chat, /api/conversations, /api/.../evaluate
│   │   ├── health.py            # /health, /ready
│   │   └── errors.py            # error handlers
│   └── services/
│       ├── chat_service.py      # RAG pipeline orchestrator
│       ├── pdf_service.py       # PDF extraction + semantic chunking
│       ├── vector_store.py      # Hybrid search (dense + sparse + RRF)
│       ├── reranker_service.py  # Qwen3-VL cross-encoder reranker
│       ├── eval_service.py      # RAGAS evaluation (faithfulness, relevancy, precision)
│       ├── llm_service.py       # Ollama wrapper
│       ├── conversation_service.py  # in-memory store
│       └── exceptions.py
├── wsgi.py
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Pipeline

```
PDF -> semantic chunk -> upsert to dense + sparse indexes
Query -> hybrid search -> RRF merge -> cross-encoder rerank -> top K -> LLM
Evaluate -> RAGAS metrics on Q&A pairs (faithfulness, relevancy, context precision)
```

## Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Pinecone keys and hosts
flask --app wsgi run --debug --port 8000
```

## Docker

```bash
docker build -t chatwithpdf-backend .
docker run --rm -p 8000:8000 --env-file .env chatwithpdf-backend
```

## Configuration

All settings come from environment variables. See `.env.example` for the full list.
