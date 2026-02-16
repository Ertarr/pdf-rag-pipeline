from typing import List, Dict, Any
from dataclasses import dataclass
import fitz

@dataclass
class chunk:
    text: str
    meta: Dict[str, Any]

# def _clean_text(t: str) -> str:


def extract_pdf_doc_pages(path: str) -> List[Dict[str, any]]:
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        pages.append({"page": i, "text": doc[i].get_text("text")})
    doc.close()
    return pages

def text_extract_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, start + 1)
    return chunks

def extract_pdf_chunks(path: str, chunk_size: int = 900, overlap: int = 150) -> List[chunk]:
    pages = extract_pdf_doc_pages(path)
    chunks = []
    for page in pages:
        page_chunks = text_extract_chunks(page["text"], chunk_size, overlap)
        for j in range(len(page_chunks)):
            chunks.append(
                chunk(
                    text = page_chunks[j], 
                    meta = {"source": path, "page": page["page"], "chunk": j}
                )
            )
    return chunks

