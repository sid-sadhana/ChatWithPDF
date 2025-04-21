import os
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


from pinecone import Pinecone, ServerlessSpec
import torch

load_dotenv()
app = Flask(__name__)
CORS(app)


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    query = data.get('query')

    
    reader = PdfReader("pdf1.pdf")
    full_text = " ".join(p.extract_text() for p in reader.pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = splitter.create_documents([full_text])
    texts = [d.page_content for d in docs]

    
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = [
        {
            "id": f"vec{i}",
            "values": vec,
            "metadata": {"text": txt}
        }
        for i, (txt, vec) in enumerate(zip(texts, embedder.embed_documents(texts)), start=1)
    ]

    pc.create_index(
        name="quickstart",
        dimension=384,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index("quickstart")
    index.upsert(vectors=vectors, namespace="ns1")

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    xq = stm.encode(query).tolist()
    qs = index.query(namespace="ns1", vector=xq, top_k=10, include_metadata=True)

    relevant = " ".join(m['metadata']['text'] for m in qs['matches'])

    
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    comp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": f"answer EXACTLY based on [{relevant}]"},
            {"role": "user",   "content": query}
        ],
        temperature=0, max_tokens=8192, stream=True
    )
    response = "".join(c.choices[0].delta.content or "" for c in comp)

    
    pc.delete_index("quickstart")
    return response

if __name__ == '__main__':
    app.run(debug=True)
