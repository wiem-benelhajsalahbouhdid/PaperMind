# PaperMind

**Question answering over scientific papers, with every answer cited to its source page.**

CiteSense is a Retrieval-Augmented Generation (RAG) system. You give it a collection of research papers in PDF, ask a question in plain English, and it answers using only the content of those papers, citing the exact paper and page for each claim.

---

## Features

- Semantic search over a collection of PDFs (search by meaning, not keywords)
- Answers grounded in the papers, with citations such as `[1]`, `[2]` pointing to the file and page
- Says "I don't know" when the answer is not in the papers
- Runs fully locally with Ollama: no data leaves your computer and there is no API cost
- Chat interface built with Streamlit, showing the sources used for each answer
- Retrieval evaluation on a hand-labelled question set

---

## How it works

**Indexing** (run once, `ingest.py`)

1. Extract the text of each PDF page by page.
2. Split the text into overlapping chunks of 800 characters (150 characters of overlap).
3. Convert each chunk into a vector (embedding) with a sentence-transformers model.
4. Store all vectors in a FAISS index, and the chunk texts with their file and page in `chunks.json`.

**Answering** (for each question, `rag.py`)

1. Convert the question into a vector with the same model.
2. Retrieve the most similar chunks from the FAISS index.
3. Build a prompt that tells the LLM to answer only from these chunks and to cite them.
4. Generate the answer with a local LLM and return it with its sources.

---

## Tech stack

- **Language:** Python
- **PDF parsing:** PyMuPDF
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS
- **LLM:** Llama 3.2 3B through Ollama (OpenAI-compatible API)
- **Interface:** Streamlit

---

## Installation

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/wiem-benelhajsalahbouhdid/citesense.git
   cd citesense
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Linux / macOS
   pip install -r requirements.txt
   ```

2. Install [Ollama](https://ollama.com) and download the model:

   ```bash
   ollama pull llama3.2:3b
   ```

3. Create a `.env` file in the project folder:

   ```env
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=llama3.2:3b
   ```

   To use the OpenAI API instead, change these three lines. No code change is needed.

No GPU is required. Everything runs on a normal laptop CPU.

---

## Usage

1. Put your PDF files in `data/papers/`.
2. Build the index:

   ```bash
   python ingest.py
   ```

3. Launch the interface:

   ```bash
   streamlit run app.py
   ```

---

## Evaluation

Retrieval is evaluated separately from generation, because if the right passage is not retrieved, the LLM cannot answer correctly.

- **Test set:** hand-written questions in `eval/questions.csv`, each labelled with the paper and page that contain the answer.
- **Hit@k:** percentage of questions where the correct page appears in the top-k results.
- **MRR (Mean Reciprocal Rank):** rewards finding the correct page near the top (rank 1 = 1, rank 2 = 0.5, miss = 0).

```bash
python evaluate.py      # top-4
python evaluate.py 8    # top-8
```

---

## Project structure

```
citesense/
├── data/papers/          # PDF files (not committed)
├── index/                # generated FAISS index and chunks.json (not committed)
├── eval/
│   └── questions.csv     # evaluation questions
├── .streamlit/
│   └── config.toml       # interface theme
├── ingest.py             # PDFs -> chunks -> embeddings -> FAISS index
├── rag.py                # retrieval and answer generation
├── app.py                # Streamlit interface
├── evaluate.py           # retrieval evaluation
├── requirements.txt
└── .env                  # LLM settings (not committed)
```

---

## Dataset

Open-access papers on federated learning, including:

- K. D. Reddy and T. R. Gadekallu, _A Comprehensive Survey on Federated Learning Techniques for Healthcare Informatics_, Computational Intelligence and Neuroscience, 2023. [DOI](https://doi.org/10.1155/2023/8393990)

---

## Limitations

- PDF extraction quality strongly affects results: some extractors glue words together, which hurts the embeddings.
- Repeated page footers and reference pages add noise to the index.
- The system always returns k chunks, even when none of them is relevant.
- An answer split across two pages may be partly missed.
- A small local LLM does not always follow the instruction to use only the sources.

## Next steps

- Remove page footers and skip reference pages during ingestion
- Ignore chunks below a minimum similarity score
- Hybrid search (keywords + vectors) and re-ranking
- Evaluate answer quality, not only retrieval
- Expose the pipeline through a FastAPI endpoint

---

## Author

**Wiem Ben El Haj Salah Bouhdid**, engineering student in Data Science and Computer Vision at ENSI, Tunisia.
[GitHub](https://github.com/wiem-benelhajsalahbouhdid)
