import os
import uuid
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise RuntimeError("PINECONE_API_KEY and GROQ_API_KEY must be set in .env")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
NAMESPACE_FILE = 'namespaces.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"


if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)


def load_namespaces():
    if os.path.exists(NAMESPACE_FILE):
        with open(NAMESPACE_FILE, 'r') as f:
            return json.load(f)
    else:
        
        with open(NAMESPACE_FILE, 'w') as f:
            json.dump({}, f)
        return {}


def save_namespaces(ns_map):
    with open(NAMESPACE_FILE, 'w') as f:
        json.dump(ns_map, f)


def cleanup_namespaces():
    ns_map = load_namespaces()
    now = datetime.utcnow()
    removed = []
    for ns, ts in list(ns_map.items()):
        created = datetime.fromisoformat(ts)
        if now - created > timedelta(minutes=2):
            index.delete(delete_all=True, namespace=ns)
            removed.append(ns)
            del ns_map[ns]
    if removed:
        save_namespaces(ns_map)

@app.route('/api/generate', methods=['POST'])
def generate():
    cleanup_namespaces()

    query = request.form.get('query')
    file = request.files.get('file')
    if not query or not file:
        return jsonify({"error": "Query and PDF file are required"}), 400

    filename = secure_filename(file.filename)
    file_uuid = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_uuid}_{filename}")
    file.save(filepath)
    namespace = file_uuid

    ns_map = load_namespaces()
    ns_map[namespace] = datetime.utcnow().isoformat()
    save_namespaces(ns_map)

    reader = PdfReader(filepath)
    full_text = " ".join(p.extract_text() or "" for p in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = splitter.create_documents([full_text])
    texts = [d.page_content for d in docs]
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embs = embedder.embed_documents(texts)
    vectors = [
        {"id": str(uuid.uuid4()), "values": vec.tolist() if hasattr(vec, 'tolist') else vec, "metadata": {"text": txt}}
        for txt, vec in zip(texts, embs)
    ]
    index.upsert(vectors=vectors, namespace=namespace)

    
    import time
    time.sleep(1.5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    xq = stm.encode(query).tolist()

    
    qs = index.query(vector=xq, namespace=namespace, top_k=10, include_metadata=True)
    if 'matches' not in qs or not qs['matches']:
        return jsonify({"error": "No relevant content found for the query."}), 404

    relevant = " ".join(m['metadata']['text'] for m in qs['matches'])

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    comp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": f"answer EXACTLY based on [{relevant}]"},
            {"role": "user", "content": query}
        ],
        temperature=0, max_tokens=8192, stream=True
    )
    response = "".join(c.choices[0].delta.content or "" for c in comp)

    try:
        os.remove(filepath)
    except OSError:
        pass

    return response

if __name__ == '__main__':
    app.run(debug=True)
