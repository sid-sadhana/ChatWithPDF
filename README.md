# ChatWithPDF

Full-stack RAG app that lets users upload a PDF and chat with it. Uses hybrid search (dense + sparse), reciprocal rank fusion, cross-encoder reranking, and a local LLM.

```
                              ┌───────────────────┐
┌────────────┐    /api        │     Backend       │
│  Frontend  │───────────────>│    Flask API      │
│ React + TW │                │                   │
│ + Framer   │                │  ┌─────────────┐  │     ┌──────────────────┐
└────────────┘                │  │ Semantic     │  │     │    Pinecone      │
                              │  │ Chunking     │  │     │                  │
                              │  └──────┬───────┘  │     │  Dense Index     │
                              │         │          │────>│  (llama-text-    │
                              │  ┌──────▼───────┐  │     │   embed-v2)     │
                              │  │ Hybrid Search│  │     │                  │
                              │  │ Dense+Sparse │  │────>│  Sparse Index   │
                              │  └──────┬───────┘  │     │  (pinecone-     │
                              │         │          │     │   sparse-v0)    │
                              │  ┌──────▼───────┐  │     └──────────────────┘
                              │  │ RRF Fusion   │  │
                              │  └──────┬───────┘  │     ┌──────────────────┐
                              │         │          │     │  Ollama (local)  │
                              │  ┌──────▼───────┐  │────>│  qwen3.5:2b      │
                              │  │   Reranker   │  │     └──────────────────┘
                              │  │ Qwen3-VL-2B  │  │
                              │  └──────┬───────┘  │
                              │         │          │
                              │  ┌──────▼───────┐  │
                              │  │  LLM Answer  │  │
                              │  └──────────────┘  │
                              └───────────────────┘
```

## RAG Pipeline

1. **Upload** PDF -> extract text -> semantic chunking (by sections/headings/paragraphs)
2. **Index** chunks into both Pinecone dense + sparse indexes (server-side embeddings)
3. **Query** -> hybrid search (dense + sparse) -> Reciprocal Rank Fusion -> Qwen3 cross-encoder reranking -> top 5 chunks
4. **Generate** answer with Ollama using retrieved context

## Retrieval Stages (visible in UI)

Each query shows results from all four retrieval stages in tabs:

| Stage | Method | Purpose |
| --- | --- | --- |
| **Dense** | llama-text-embed-v2 | Semantic similarity search |
| **Sparse** | pinecone-sparse-english-v0 | Keyword/lexical matching |
| **Fused (RRF)** | Reciprocal Rank Fusion (K=60, alpha=0.7) | Merge dense + sparse rankings |
| **Reranked** | Qwen3-VL-Reranker-2B cross-encoder | Pointwise relevance scoring |

## Stack

- **Frontend** React, Tailwind CSS, Framer Motion
- **Backend** Flask, Gunicorn
- **Dense Embeddings** Pinecone Inference API (llama-text-embed-v2)
- **Sparse Embeddings** Pinecone Inference API (pinecone-sparse-english-v0)
- **Reranker** Qwen3-VL-Reranker-2B (local cross-encoder)
- **LLM** Ollama (qwen3.5:2b)
- **Vector DB** Pinecone (namespace per conversation)

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
│   │       ├── reranker_service.py
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
└── docker-compose.yml
```

## Prerequisites

- [Ollama](https://ollama.com/) installed locally
- A [Pinecone](https://www.pinecone.io/) account + API key
- Python 3.9+ and Node.js 18+

## Quick start

### 1. Ollama model

```bash
ollama pull qwen3.5:2b
```

### 2. Pinecone indexes

Create two indexes in the Pinecone dashboard:
- **Dense index** with integrated model `llama-text-embed-v2`
- **Sparse index** with integrated model `pinecone-sparse-english-v0`

### 3. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in PINECONE_API_KEY, PINECONE_DENSE_HOST, PINECONE_SPARSE_HOST
flask --app wsgi run --debug --port 8000
```

First run downloads the Qwen3-VL-Reranker-2B model (~4GB).

### 4. Frontend

```bash
cd frontend
npm install
npm start
```

Frontend on http://localhost:3000, backend on http://localhost:8000.

## Docker

```bash
cp backend/.env.example backend/.env
# fill in Pinecone keys and hosts
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend: http://localhost:8000/health

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PINECONE_API_KEY` | -- | Pinecone credentials (required) |
| `PINECONE_DENSE_HOST` | -- | Dense index host URL (required) |
| `PINECONE_SPARSE_HOST` | -- | Sparse index host URL (required) |
| `HYBRID_ALPHA` | `0.7` | Dense vs sparse weight (0-1) |
| `OLLAMA_MODEL` | `qwen3.5:2b` | LLM model |
| `RERANKER_MODEL` | `Qwen/Qwen3-VL-Reranker-2B` | Cross-encoder reranker |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Final results after reranking |

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
