import os
import faiss
import json
from sentence_transformers import SentenceTransformer
import ollama

INDEX_DIR = "index"
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_TEXT_PATH = os.path.join(INDEX_DIR, "chunks_text.json")

def load_index():
    index = faiss.read_index(FAISS_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(CHUNKS_TEXT_PATH, "r", encoding="utf-8") as f:
        texts = json.load(f)

    return index, meta, texts

def generate_answer(question: str, contexts: list[dict]):
    """
    contexts: liste de dicts avec keys: text, source, page
    """

    ctx = "\n\n".join(
        [f"[{i+1}] source: {c['source']} (p.{c['page']})  {c['text']}" for i,c in enumerate(contexts)]
    )

    system = (
        "Réponds uniquement à partir des extraits fournis. "
        "Si l'information n'est pas dans les extraits, dis: 'Je ne trouve pas cette information dans les documents.' "
        "A la fin donne les sources sous forme [1], [2], etc... "
    )

    prompt = f"""Question: {question} \n\nExtraits: {ctx} \n\nRéponse (en français, concise, avec citations [1], [2]...):"""

    resp = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    return resp["message"]["content"]

def main():
    q = input("Question: ").strip()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")

    index, meta, texts = load_index()
    k = 5

    scores, ids = index.search(q_emb, k)

    contexts = []

    for (score, id) in zip(scores[0], ids[0]):
        
        m = meta[id]
        snippet = texts[id]

        contexts.append({
                "text": snippet,
                "source": m["source"],
                "page": m["page"],
                "score": float(score)
            
        })

    answer = generate_answer(q, contexts)

    print("\n\tRéponse:\t\n")
    print(answer)

    print("\n\tSources:\t\n")
    for i, c in enumerate(contexts, start=1):
        print(f"""[{i}] {c["source"]}  (page {c["page"]}) score={c['score']:.3f}""")
        

if __name__ == "__main__":
    main()