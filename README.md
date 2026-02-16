# PDF RAG Pipeline

Petit projet RAG (Retrieval-Augmented Generation) sur des documents PDF :
- Extraction du texte depuis des PDFs (avec pages)
- Découpage en chunks
- Embeddings (Sentence-Transformers)
- Indexation + recherche vectorielle (FAISS)
- Retour de la réponse à partir des passages pertinents avec citations (source/page)
- Interface Streamlit

---

## Structure du projet

```
pdf-rag-pipeline/
  data/
    docs/            # <-- mets tes PDFs ici
  index/             # <-- généré automatiquement (FAISS + metadata)
  src/
    ingest.py        # construit l'index
    pdf_utils.py     # extraction + chunking
    rag.py           # question -> réponse (à partir des top-k passages) + liste des sources [1], [2]…
  app.py             # interface Streamlit
  requirements.txt
  README.md
```

---

## Pré-requis
- Python 3.10+ (3.11 recommandé)

---

## Installation

### Option A — Installer via requirements.txt (recommandé)
```bash
pip install -r requirements.txt
```

### Option B — Installer manuellement
```bash
pip install pymupdf sentence-transformers faiss-cpu numpy tqdm streamlit ollama
```

### Génération avec Mistral (Ollama)
- Installer Ollama et télécharger le modèle :
```bash
ollama pull mistral
```
Ollama doit être installé et accessible en local.

- Avoir la liste des modèles téléchargés :
```bash
ollama list
```

---

## Ajouter des PDFs

1. Crée (ou vérifie) le dossier :
```
data/docs/
```

2. Mets tes fichiers `.pdf` dedans, par exemple :
```
data/docs/rapport.pdf
data/docs/specifications.pdf
data/docs/cours.pdf
```

---

## Construire l’index (ingestion)

Cette étape :
- lit tous les PDFs dans `data/docs/`
- découpe en chunks
- calcule les embeddings
- crée l’index FAISS dans `index/`

Commande :
```bash
python src/ingest.py
```

À la fin, tu dois voir des fichiers dans `index/` (exemples) :
- `faiss.index`
- `chunks_meta.json`
- `chunks_text.json`

---

## Poser une question (recherche + citations)

Lance le mode CLI :
```bash
python src/rag.py
```

Tu tapes une question et le script affiche la réponse à partir des meilleurs passages + les citations (sources/pages).

Exemple de question : "Résume les points clés et cite les pages."

---

## Lancer l’app Streamlit

```bash
streamlit run app.py
```

---

## Notes / Dépannage

### Problème avec FAISS (souvent sur Windows)
Si `faiss-cpu` ne s’installe pas :
- essayer une autre version de Python (souvent 3.10/3.11 marche mieux)
- ou remplacer FAISS par une base vectorielle type Chroma.

```bash
pip install chromadb
```

Alternative possible : Chroma (à intégrer dans le code si besoin).

---

