import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("index/index.faiss")
chunks = json.load(open("index/chunks.json", encoding="utf-8"))

def retrieve(question, k=4):
       q = model.encode([question], normalize_embeddings=True)
       scores, ids = index.search(np.asarray(q, dtype="float32"), k)
       return [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0])]

if __name__ == "__main__":
       for r in retrieve("What is federated averaging?"):
           print(round(r["score"], 3), r["file"], "p.", r["page"], "|", r["text"][:150])