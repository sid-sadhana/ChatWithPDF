# Backend — ChatWithPDF API

Flask API that ingests PDFs, embeds chunks via Pinecone Inference API, stores them in Pinecone, and generates answers using a local Ollama LLM.

## Layout

```
backend/
├── app/
│   ├── __init__.py          # app factory
│   ├── config.py            # env-driven config
│   ├── api/
│   │   ├── routes.py        # /api/upload, /api/chat, /api/conversations
│   │   ├── health.py        # /health, /ready
│   │   └── errors.py        # error handlers
│   └── services/
│       ├── chat_service.py  # RAG pipeline
│       ├── pdf_service.py   # PDF text extraction + chunking
│       ├── vector_store.py  # Pinecone (integrated embeddings)
│       ├── llm_service.py   # Ollama wrapper
│       ├── conversation_service.py  # in-memory store
│       └── exceptions.py
├── wsgi.py
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Local development

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY and PINECONE_INDEX_HOST
flask --app wsgi run --debug --port 8000
```

## Docker

```bash
docker build -t chatwithpdf-backend .
docker run --rm -p 8000:8000 --env-file .env chatwithpdf-backend
```

## Configuration

All settings come from environment variables. See `.env.example` for the full list.
