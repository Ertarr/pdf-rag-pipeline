import os
import glob
from tqdm import tqdm
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

from pdf_utils import extract_pdf_chunks

INDEX_DIR = "index"
DOCS_DIR = "data/docs"
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_TEXT_PATH = os.path.join(INDEX_DIR, "chunks_text.json")

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    pdf_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))

    if not pdf_paths:
        raise SystemExit(f"Aucun PDF trouvé dans {DOCS_DIR}")
    
    chunks_text = []
    chunks_meta = []

    for path in tqdm(pdf_paths, desc="Parsing PDFs"):
        chunks = extract_pdf_chunks(path)

        for chunk in chunks:
            chunks_text.append(chunk.text)
            chunks_meta.append(chunk.meta)


    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(chunks_text, batch_size = 64, show_progress_bar = True, normalize_embeddings = True)
    emb = np.asarray(emb, dtype = "float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "w", encoding = "utf-8") as f:
        json.dump(chunks_meta, f, ensure_ascii=False, indent=2)
    with open(CHUNKS_TEXT_PATH, "w", encoding = "utf-8") as f:
        json.dump(chunks_text, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(chunks_text)} chunks indexés.")
    print(f"FAISS: {FAISS_PATH}")
    print(f"META : {META_PATH}")
    print(f"META : {CHUNKS_TEXT_PATH}")


if __name__ == "__main__":
    main()