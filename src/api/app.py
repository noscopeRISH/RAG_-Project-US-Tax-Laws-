"""FastAPI application for the minimal hybrid RAG search system."""

import os
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


# Global uuid monkey-patch to prevent NameError inside Pydantic runtime
import builtins
import uuid
builtins.uuid = uuid
builtins.UUID = uuid.UUID

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.hybrid_retriever import HybridRetriever
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse, Citation

# Global dictionary to keep singletons alive
_RAG_PIPELINE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan event handler that loads documents and builds index once on startup."""
    print("[DEBUG] FastAPI Lifespan: Initializing RAG resources...", flush=True)
    
    # 1. Initialize Document Processor
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    sources = ["data"]
    retriever = HybridRetriever()
    
    # 2. Check if cache files exist and MD5 corpus hash matches
    corpus_dir = "data"
    current_hash = retriever._calculate_corpus_hash(corpus_dir)
    faiss_dir = Path(retriever.faiss_path)
    bm25_file = Path(retriever.bm25_path)
    chunks_file = Path(retriever.chunks_path)
    hash_file = Path(retriever.hash_path)
    
    cache_valid = (
        faiss_dir.exists()
        and bm25_file.exists()
        and chunks_file.exists()
        and hash_file.exists()
        and current_hash
    )
    
    if cache_valid:
        try:
            stored_hash = hash_file.read_text().strip()
            if stored_hash == current_hash:
                print("[DEBUG] FastAPI Lifespan: Cache is valid. Skipping document parsing.", flush=True)
                retriever.create_index([])  # Loads existing index from disk since hash matches
                _RAG_PIPELINE["retriever"] = retriever
                _RAG_PIPELINE["chunks_count"] = len(retriever.chunks)
                print(f"[DEBUG] FastAPI Lifespan: RAG initialization finished from cache. ({len(retriever.chunks)} chunks loaded)", flush=True)
                yield
                _RAG_PIPELINE.clear()
                print("[DEBUG] FastAPI Lifespan: RAG resources cleaned up.", flush=True)
                return
        except Exception as e:
            print(f"[DEBUG] FastAPI Lifespan: Failed to load index from cache: {e}. Rebuilding...", flush=True)
            
    # 3. Load documents (fallback)
    print("[DEBUG] FastAPI Lifespan: Cache invalid or missing. Starting document ingestion...", flush=True)
    documents = doc_processor.process_urls(sources)
    
    # 4. Build Hybrid Index
    retriever.create_index(documents)
    
    # Cache inside the global state
    _RAG_PIPELINE["retriever"] = retriever
    _RAG_PIPELINE["chunks_count"] = len(documents)
    
    print(f"[DEBUG] FastAPI Lifespan: RAG initialization finished. ({len(documents)} chunks loaded)", flush=True)
    yield
    # Cleanup on shutdown
    _RAG_PIPELINE.clear()
    print("[DEBUG] FastAPI Lifespan: RAG resources cleaned up.", flush=True)


app = FastAPI(
    title="Tax & Legal Minimal Hybrid RAG API",
    description="Minimal RAG system using FAISS, BM25, and Mistral AI",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the static vanilla HTML search UI."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="HTML template file not found.")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@app.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    """Executes hybrid retrieval, merges via RRF, calls Mistral AI, and validates citations."""
    retriever: HybridRetriever = _RAG_PIPELINE.get("retriever")
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG system not initialized yet.")
    
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        # 1. Retrieve top 5 matching chunks via RRF Hybrid Search
        top_chunks = retriever.retrieve(question, k=5)
        
        # 2. Format retrieved chunks into context
        context_parts = []
        for idx, doc in enumerate(top_chunks, 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            src = os.path.basename(meta.get("source", f"doc_{idx}"))
            pg = meta.get("page", 0) + 1
            context_parts.append(f"[{idx}] [Source: {src}, Page: {pg}]\n{doc.page_content}")
        
        context_str = "\n\n".join(context_parts)
        
        # 3. Setup LLM and Prompt
        llm = Config.get_llm()
        prompt = f"""You are a helpful, professional US tax and legal assistant. Use ONLY the retrieved source context below to answer the user's question.
If the answer is not in the context, state that you cannot find the answer and leave the citations list empty.

CRITICAL REQUIREMENTS:
1. Ground every statement in your answer in the retrieved sources.
2. For every fact or statement you use, you must cite the source. The source name and page number are provided at the start of each retrieved passage in the format: '[Source: <filename>, Page: <page_number>]'.
3. If the retrieved documents do not contain the answer, or are unrelated to the question, you must output an answer saying you cannot find the answer in the loaded documents, and leave the citations list empty.
4. You must format your final output strictly as a single JSON object. Do NOT wrap it in markdown code blocks like ```json ... ```. Do NOT write any conversational text outside the JSON object.

The JSON object must have exactly these keys:
{{
  "answer": "A detailed explanation answering the question...",
  "citations": [
    {{"source": "filename.pdf", "page": 1}},
    {{"source": "filename2.pdf", "page": 3}}
  ]
}}

Context:
{context_str}

Question: {question}"""

        # 4. Generate answer from Mistral AI
        response = llm.invoke(prompt)
        raw_answer = response.content if hasattr(response, "content") else str(response)
        
        # 5. Parse JSON response
        clean_content = raw_answer.strip()
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\n", "", clean_content)
            clean_content = re.sub(r"\n```$", "", clean_content)
            clean_content = clean_content.strip()
            
        parsed_answer = raw_answer
        parsed_citations = []
        try:
            parsed = json.loads(clean_content)
            parsed_answer = parsed.get("answer", raw_answer)
            parsed_citations = parsed.get("citations", [])
        except Exception:
            # Fallback: regex search for pattern "Source: X, Page: Y"
            found = re.findall(r"Source:\s*([^\s,]+),\s*Page:\s*(\d+)", raw_answer)
            if found:
                parsed_citations = [{"source": src, "page": int(pg)} for src, pg in found]
        
        # 6. Citations Grounding Verification (cross-check against actually retrieved chunks)
        valid_citations_set = set()
        for doc in top_chunks:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            src = os.path.basename(meta.get("source", ""))
            pg = meta.get("page", 0) + 1
            if src:
                valid_citations_set.add((src.lower(), pg))
        
        verified_citations = []
        for cite in parsed_citations:
            source = cite.get("source", "")
            page = cite.get("page")
            if source and page is not None:
                source_base = os.path.basename(source)
                try:
                    page_int = int(page)
                    # Case-insensitive comparison of filename
                    if (source_base.lower(), page_int) in valid_citations_set:
                        verified_citations.append(
                            Citation(source=source_base, page=page_int)
                        )
                except (ValueError, TypeError):
                    pass
                    
        return QueryResponse(
            answer=parsed_answer,
            citations=verified_citations
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Query Execution Error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns the health status and the total count of loaded document chunks."""
    chunks_count = _RAG_PIPELINE.get("chunks_count")
    if chunks_count is None:
        return HealthResponse(status="initializing", chunks_count=0)
    return HealthResponse(status="healthy", chunks_count=chunks_count)
