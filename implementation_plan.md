# Implementation Plan - Minimal RAG with Mistral AI, Hybrid Search & HTML UI

This plan outlines the restructuring of the project to remove LangGraph (no graph RAG), load and chunk documents (~500-800 tokens with overlap), generate embeddings using Hugging Face `sentence-transformers`, store in a FAISS vector DB, build a BM25 keyword index over matching chunk IDs, run RRF hybrid search, generate answers using Mistral AI API, and serve a static HTML UI page directly from the FastAPI root endpoint.

---

## File Tree Preview

```
RAG_1/
├── src/
│   ├── api/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── templates/
│   │       └── index.html
│   ├── config/
│   │   └── config.py
│   ├── document_ingestion/
│   │   └── document_processor.py
│   └── vectorstore/
│       └── hybrid_retriever.py
├── deprecated/
│   ├── graph_builder.py
│   ├── nodes.py
│   ├── reactnode.py
│   ├── rag_state.py
│   └── dependencies.py
├── api_main.py
├── pyproject.toml
└── requirements.txt
```

---

## Refined Core Designs

### 1. Ingestion & Chunking
* Load PDFs recursively from `data/` using `PyPDFLoader` individually (with terminal progress prints).
* Chunk text using `RecursiveCharacterTextSplitter` with a chunk size representing ~500-800 tokens (e.g. 2000-3000 characters) and overlap of ~10% (e.g. 200-300 characters).

### 2. Hybrid Search (FAISS + BM25 + RRF)
* **Vector Index**: FAISS vector store on disk (`data/faiss_index`) using `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`.
* **Keyword Index**: BM25 index built over the exact same chunks.
* **Corpus Hash & Locking**: Retain corpus hashing and atomic file locking to prevent race conditions during startup.
* **RRF Merging**: Implement Reciprocal Rank Fusion (RRF) to score and sort chunks retrieved by both vector and keyword queries:
  $$RRF\_Score(d) = \sum_{r \in R} \frac{1}{60 + rank(d, r)}$$

### 3. Generation using Mistral AI API
* Use `ChatMistralAI` model (`mistral-large-latest`) configured with `MISTRAL_API_KEY`.
* Prompt the model to ground the answer in the retrieved context chunks and output the answer and a list of citation source document names.
* Validate that citations actually exist in the retrieved document chunks.

### 4. FastAPI Backend & HTML UI
* `/` — serves static `index.html` via `HTMLResponse`.
* `/query` — receives question, runs hybrid RRF search, calls Mistral AI, and returns answer + verified citations.
* `/health` — returns status and the total count of loaded document chunks.
* `index.html` — single static HTML page with vanilla CSS and JavaScript to query the API and render answers and sources.

---

## Deprecation & Cleanup
* Move `graph_builder.py`, `nodes.py`, `reactnode.py`, `rag_state.py`, and `dependencies.py` to the `deprecated/` folder.

---

## Verification Plan
* Run `api_main.py` using Python.
* Open `http://localhost:8000/` and run query searches to verify the UI, answer, and sources list.
