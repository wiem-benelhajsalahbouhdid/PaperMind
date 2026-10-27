import json
import os
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer


def read_pdf(path):
    pages = []
    for i, page in enumerate(pymupdf.open(path)):
        text = " ".join(page.get_text().split())
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


chunks = []
for pdf in sorted(Path("data/papers").glob("*.pdf")):
    for page_num, text in read_pdf(pdf):
        for piece in chunk_text(text):
            chunks.append({"file": pdf.name, "page": page_num, "text": piece})
    print("Read", pdf.name)
print("Total chunks:", len(chunks))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True)
vectors = np.asarray(vectors, dtype="float32")

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/index.faiss")
with open("index/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)
print("Saved", index.ntotal, "vectors")