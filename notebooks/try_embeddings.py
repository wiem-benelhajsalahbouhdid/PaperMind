from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentences = [
       "Federated learning trains models without sharing raw data.",
       "Clients keep their data locally during training.",
       "The weather in Tunis is sunny today.", ]
vectors = model.encode(sentences, normalize_embeddings=True)
print(vectors.shape)
print(vectors @ vectors.T)
