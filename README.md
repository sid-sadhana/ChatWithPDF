# ChatWithPDF

Full-stack AI app that lets users upload a PDF and chat with it using naive RAG. Documents are chunked, embedded via Pinecone's Inference API (llama-text-embed-v2), stored in Pinecone, and answered by a local Ollama LLM.

```
                         ┌──────────────────┐
┌────────────┐   /api    │     Backend      │     ┌──────────────┐
│  Frontend  │──────────▶│   Flask API      │────▶│   Pinecone   │
│ React+TW   │           │                  │     │  Vector DB + │
└────────────┘           │                  │     │  Embeddings  │
                         │                  │     └──────────────┘
                         │                  │     ┌──────────────┐
                         │                  │────▶│ Ollama (LLM) │
                         └──────────────────┘     └──────────────┘
```

## Stack

- **Frontend** React, Tailwind CSS, Framer Motion
- **Backend** Flask, Gunicorn
- **Embeddings** Pinecone Inference API (llama-text-embed-v2, server-side)
- **Vector DB** Pinecone (namespace per conversation)
- **LLM** Ollama (qwen3.5:2b, local)
- **Conversations** In-memory

## Repository layout

```
ChatWithPDF/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── routes.py
│   │   │   ├── health.py
│   │   │   └── errors.py
│   │   └── services/
│   │       ├── chat_service.py
│   │       ├── pdf_service.py
│   │       ├── vector_store.py
│   │       ├── llm_service.py
│   │       ├── conversation_service.py
│   │       └── exceptions.py
│   ├── wsgi.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── docker-compose.override.yml
└── .env.example
```

## Prerequisites

- [Ollama](https://ollama.com/) installed locally
- A [Pinecone](https://www.pinecone.io/) account + API key
- Python 3.9+ and Node.js 18+

## Quick start

### 1. Pull Ollama model

```bash
ollama pull qwen3.5:2b
```

### 2. Create Pinecone index

Create an index named `ragora` in the Pinecone dashboard with integrated embedding model `llama-text-embed-v2`.

### 3. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY and PINECONE_INDEX_HOST
flask --app wsgi run --debug --port 8000
```

### 4. Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on http://localhost:3000, backend on http://localhost:8000.

## Docker

```bash
cp backend/.env.example backend/.env
# fill in PINECONE_API_KEY and PINECONE_INDEX_HOST
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend: http://localhost:8000/health

## Configuration

All settings are environment-driven. See `backend/.env.example`:

| Variable | Default | Description |
| --- | --- | --- |
| `PINECONE_API_KEY` | — | Pinecone credentials (required) |
| `PINECONE_INDEX_HOST` | — | Pinecone index host URL (required) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3.5:2b` | LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Top-k results retrieved |

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/upload` | Upload PDF (multipart) |
| `POST` | `/api/chat` | Send message `{conversation_id, query}` |
| `GET` | `/api/conversations` | List all conversations |
| `GET` | `/api/conversations/:id` | Get conversation with messages |
| `DELETE` | `/api/conversations/:id` | Delete a conversation |
| `DELETE` | `/api/conversations` | Delete all data |
| `GET` | `/health` | Liveness probe |

## License

MIT
