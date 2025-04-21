from flask import Flask, request
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import torch
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)
CORS(app) 