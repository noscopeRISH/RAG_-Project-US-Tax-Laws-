"""Pre-builds the RAG indexes during Docker image creation to eliminate cold-start delays at runtime."""

import os
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.hybrid_retriever import HybridRetriever
from src.config.config import Config

def main():
    print("[BUILD-TIME] Starting RAG index construction...", flush=True)
    
    # 1. Initialize Document Processor
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    # 2. Ingest and split documents
    sources = ["data"]
    print(f"[BUILD-TIME] Ingesting documents from sources: {sources}", flush=True)
    documents = doc_processor.process_urls(sources)
    
    # 3. Build and serialize FAISS & BM25 indexes
    print(f"[BUILD-TIME] Generating hybrid indexes for {len(documents)} chunks...", flush=True)
    retriever = HybridRetriever()
    retriever.create_index(documents)
    
    print("[BUILD-TIME] RAG indexes successfully built and cached!", flush=True)

if __name__ == "__main__":
    main()
