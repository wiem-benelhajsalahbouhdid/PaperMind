from pypdf import PdfReader 
import json, os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

reader=PdfReader("data/papers/SURVEY2.pdf")
print("pages:",len(reader.pages))
text= reader.pages[0].extract_text()
print(text[ :500])
text="".join(text.split())
def read_pdf(path):
    pages=[]
    for i ,page in enumerate(PdfReader(path).pages):
        text="".join((page.extract_text()or "").split())
        if text:
            pages.append((i+1 , text ))
    return pages 

def chunk_text ( text , size=800, overlap=150):
    chunks=[]
    start = 0 
    while start < len(text):
        chunks.append(text[start:start+size])
        start+= size-overlap 
    return chunks 
print(len(chunk_text("a"*2000)))
from pathlib import Path

chunks = []
for pdf in sorted(Path("data/papers").glob("*.pdf")):
       for page_num, text in read_pdf(pdf):
           for piece in chunk_text(text):
               chunks.append({"file": pdf.name, "page": page_num, "text": piece})
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
