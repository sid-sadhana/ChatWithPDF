import os
import json
from datetime import datetime
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
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
NAMESPACE_LOG = "namespaces.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "quickstart"

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

def load_namespace_log():
    if not os.path.exists(NAMESPACE_LOG):
        return {}
    with open(NAMESPACE_LOG, "r") as f:
        return json.load(f)

def save_namespace_log(log):
    with open(NAMESPACE_LOG, "w") as f:
        json.dump(log, f, indent=2)

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        query = request.form.get('query')
        file = request.files.get('file')

        if not query or not file:
            return jsonify({"error": "Query or file not provided"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
    except Exception as e:
        return jsonify({"error": f"Invalid request format: {str(e)}"}), 400

    namespace = os.path.splitext(os.path.basename(filepath))[0]
    namespace_log = load_namespace_log()

    if namespace not in namespace_log:
        try:
            reader = PdfReader(filepath)
            full_text = " ".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500

        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
            docs = splitter.create_documents([full_text])
            texts = [d.page_content for d in docs]

            embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectors = [
                {
                    "id": f"{namespace}_vec{i}",
                    "values": vec,
                    "metadata": {"text": txt}
                }
                for i, (txt, vec) in enumerate(zip(texts, embedder.embed_documents(texts)), start=1)
            ]

            index = pc.Index(INDEX_NAME)
            index.upsert(vectors=vectors, namespace=namespace)

            namespace_log[namespace] = datetime.utcnow().isoformat()
            save_namespace_log(namespace_log)
        except Exception as e:
            return jsonify({"error": f"Vector processing/upsert failed: {str(e)}"}), 500

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        stm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        xq = stm.encode(query).tolist()

        index = pc.Index(INDEX_NAME)
        qs = index.query(namespace=namespace, vector=xq, top_k=10, include_metadata=True)
        relevant = " ".join(m['metadata']['text'] for m in qs['matches'])
    except Exception as e:
        return jsonify({"error": f"Embedding/query failed: {str(e)}"}), 500

    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        comp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"answer EXACTLY based on '''{relevant}]'''"},
                {"role": "user",   "content": query}
            ],
            temperature=0, max_tokens=8192, stream=True
        )
        response = "".join(c.choices[0].delta.content or "" for c in comp)
    except Exception as e:
        return jsonify({"error": f"LLM processing failed: {str(e)}"}), 500

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
