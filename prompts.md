# Prompts Used

This document records the full sequence of prompts used to build and extend the Agentic RAG project for US Tax & Legal documents.

---

## Prompt 1 — Extending the existing Agentic RAG project (Hybrid Search, Citations, FastAPI)

```
I have an existing modular Agentic RAG project (LangChain + LangGraph + FAISS + 
Streamlit) with this structure:
RAG_1/
├── src/
│   ├── config/config.py
│   ├── document_ingestion/document_processor.py
│   ├── vectorstore/vectorstore.py
│   ├── nodes/nodes.py
│   ├── graph_builder/graph_builder.py
│   └── state/rag_state.py
├── data/
└── streamlit_app.py
This is a Tax & Legal (US) RAG system over a corpus of ~100 documents 
(Acts, court judgments, and POV/commentary pieces). I need to extend it 
with three capabilities: a FastAPI backend, citation/reference support, 
and hybrid search. Please address them in this order, since each builds 
on the previous one.

PHASE 1 — Hybrid Search (retrieval layer)
Currently src/vectorstore/vectorstore.py only does dense vector search via 
FAISS. Extend it to support hybrid search:
- Add a keyword-based retriever using BM25 (e.g. via langchain's BM25Retriever) 
  built over the same document chunks
- Combine BM25 and FAISS vector retrieval using an EnsembleRetriever (or 
  equivalent weighted fusion approach), with configurable weights between 
  keyword and semantic search
- Expose a `get_hybrid_retriever(keyword_weight=0.4, vector_weight=0.6)` 
  method on the VectorStore class so the weighting can be tuned
- Keep the existing `create_retriever` and `get_retriever` methods intact 
  for backward compatibility — hybrid search should be additive, not a 
  breaking change

PHASE 2 — Citations & References (node/state layer)
Currently src/nodes/nodes.py builds an answer but discards the source 
metadata. Update this so:
- Each retrieved Document's metadata (source filename, page number) is 
  preserved through the pipeline rather than dropped
- The answer-generation prompt instructs the LLM to ground its answer in 
  the retrieved context and the final response includes a structured 
  citations list: for each fact used, the source document name and page number
- Update src/state/rag_state.py if needed to add a `citations` field 
  (list of dicts with `source` and `page` keys) alongside the existing 
  `answer` field
- The Streamlit UI's "Source Documents" expander should be updated to 
  show these citations clearly (document name + page number) rather than 
  just raw chunk text

PHASE 3 — FastAPI Backend (API layer)
Create a new module `src/api/` containing:
- `app.py` — FastAPI app with route definitions
- `schemas.py` — Pydantic request/response models, including a `Citation` 
  model (source, page) and a `QueryResponse` model (answer, citations, 
  response_time_sec)
- `dependencies.py` — startup-time singleton initialization of the document 
  processor, hybrid vector store, and graph builder using FastAPI's 
  lifespan event, so the pipeline builds once and is reused across requests

Endpoints:
- `POST /query` — accepts `{"question": str, "use_hybrid": bool = True}`, 
  runs it through GraphBuilder.run(), returns the answer with full citations
- `GET /health` — returns pipeline status and indexed document count
- `GET /documents` — returns corpus metadata (count, source filenames, 
  document categories if tracked)

Add `api_main.py` at the project root to run via uvicorn, separate from 
streamlit_app.py. Update requirements.txt / pyproject.toml with fastapi, 
uvicorn, and rank_bm25.

Across all three phases, follow the same docstring style, type hints, and 
single-responsibility class design already used in my existing modules — 
do not collapse this into one large script. Show me the full updated file 
tree first, then implement each phase's files in order.
```

---

## Prompt 2 — Requesting a detailed implementation plan and verification before execution

```
Thanks — before I approve, walk me through the implementation_plan.md in 
more detail so I can verify a few things:

1. STATE SCHEMA CHANGES
   - Show the exact updated RAGState definition with the new `citations` field. 
     Confirm whether this is a breaking change for any existing callers of 
     RAGState (e.g. does graph_builder.run() or streamlit_app.py need updates 
     beyond what we discussed?)

2. PROMPT DESIGN FOR STRUCTURED CITATIONS
   - Show me the exact prompt template you plan to use to get the LLM to 
     return structured JSON with citations. 
   - How will you handle cases where the LLM returns malformed JSON or omits 
     citations? Is there a parsing fallback so the app doesn't crash on a 
     bad LLM response?

3. HYBRID SEARCH WEIGHTING
   - Confirm the default keyword_weight/vector_weight split and explain why 
     that default makes sense for a legal/tax document corpus (where exact 
     term matches like section numbers and statute citations often matter 
     more than pure semantic similarity).
   - Will BM25Retriever be rebuilt every time create_retriever is called, 
     or cached separately from the FAISS index?

4. FASTAPI LIFESPAN INITIALIZATION
   - Confirm the FastAPI startup sequence won't duplicate work already done 
     by Streamlit — i.e., if both streamlit_app.py and api_main.py run 
     simultaneously, do they each build their own independent in-memory 
     index, or is there a shared persisted index (e.g. saved/loaded FAISS 
     index from disk) so we're not re-embedding all documents twice?

5. BACKWARD COMPATIBILITY
   - Confirm get_retriever() (vector-only) still works unchanged for any 
     existing code path, and that hybrid search is purely opt-in via 
     get_hybrid_retriever().

6. TESTING PLAN
   - Before full execution, what is your plan to test each phase in isolation 
     (e.g. test hybrid search retrieval quality on a few sample queries before 
     wiring it into the full graph; test citation JSON parsing on a few 
     responses before connecting to FastAPI)?

Once you've answered these, give me a final ordered checklist of files that 
will be created or modified in each phase, then I will approve and you can 
begin execution starting with Phase 1.
```

---

## Prompt 3 — Skipping isolation testing due to time/token constraints, proceeding to Phase 3

```
Stop the citations test — I'm almost out of time and tokens. Skip 
isolation testing for now. Proceed immediately to Phase 3: build 
src/api/dependencies.py, app.py, schemas.py, and api_main.py, wiring 
get_hybrid_retriever() into the lifespan startup as planned. I will 
test everything end-to-end once via the actual /query endpoint instead 
of separate isolation scripts. Show me all four files now, fully 
implemented, in one response.
```

---

## Prompt 4 — Building a simplified, minimal RAG backend from scratch (final deployed version)

```
Build a simple RAG backend in FastAPI for US Tax & Legal documents (~100 docs: acts, court judgments). Keep it minimal — no graph RAG, no page indexing.

Pipeline:
Ingestion script: load docs, chunk (~500-800 tokens with overlap), generate embeddings using a Hugging Face embedding model (e.g. sentence-transformers), store in a vector DB (Chroma or FAISS), and build a keyword/BM25 index over the same chunks using matching chunk IDs.
/query endpoint: accept a user question, run vector search + keyword search, merge results (reciprocal rank fusion or weighted score), take top-k chunks, then call the Mistral AI API to generate an answer with a short summary and citations referencing which source chunk/document each part came from.
/health endpoint: return status and count of loaded document chunks.
UI: a single static HTML page (vanilla JS, fetch API) served directly by FastAPI — textbox for query, submit button, results panel showing the answer plus a "Sources" list.

Use the Mistral AI API for generation and a Hugging Face sentence-transformers model for embeddings. Keep code modular: separate files for ingestion, retrieval (hybrid search), generation, and API routes.
```

---

## Notes

- Prompts 1–3 reflect an initial, more ambitious attempt to extend an existing LangChain/LangGraph-based Agentic RAG project with hybrid search, citations, and a FastAPI layer.
- Due to time and token constraints encountered mid-build (see Prompt 3), the approach was simplified.
- Prompt 4 represents the final, simplified architecture that was actually implemented, ingested, deployed, and submitted — a minimal FastAPI + Chroma/FAISS + BM25 + Mistral AI pipeline with a single static HTML UI.
