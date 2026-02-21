import os
import json
import numpy as np
import streamlit as st
import faiss
import ollama
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")
TEXTS_PATH = os.path.join(INDEX_DIR, "chunks_text.json")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"Index FAISS introuvable: {FAISS_PATH}. Lance d'abord: python src/ingest.py"
        )

    return faiss.read_index(FAISS_PATH)

@st.cache_data
def load_meta_and_texts():
    if not os.path.exists(META_PATH) or not os.path.exists(TEXTS_PATH):
        raise FileNotFoundError(
            f"Fichiers meta/texts introuvables dans {INDEX_DIR}. "
            f"Vérifie que chunks_meta.json et chunks_text.json existent."
        )
    
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        texts = json.load(f)
    return meta, texts

