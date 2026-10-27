import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("index/index.faiss")
chunks = json.load(open("index/chunks.json", encoding="utf-8"))

def retrieve(question, k=4):
       q = model.encode([question], normalize_embeddings=True)
       scores, ids = index.search(np.asarray(q, dtype="float32"), k)
       return [dict(chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0])]

load_dotenv()

def answer(question, k=4):
       sources = retrieve(question, k)
       context = "\n\n".join(f"[{n+1}] ({s['file']}, p.{s['page']}) {s['text']}"
                             for n, s in enumerate(sources))
       prompt = ("Answer using ONLY the sources below. Cite them like [1]. "
                 "If the answer is not in the sources, say you don't know.\n\n"
                 f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:")
       client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
       resp = client.chat.completions.create(
           model=os.getenv("LLM_MODEL"),
           messages=[{"role": "user", "content": prompt}],
           temperature=0)
       return resp.choices[0].message.content, sources


if __name__ == "__main__":
       for r in retrieve("What is federated averaging?"):
           print(round(r["score"], 3), r["file"], "p.", r["page"], "|", r["text"][:150])




   