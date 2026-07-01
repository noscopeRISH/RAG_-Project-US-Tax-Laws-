"""hybrid retriever module for document embedding, keyword indexing, and RRF merging"""

import os
import pickle
import hashlib
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    """Manages FAISS vector search and BM25 keyword search, merging results via RRF."""
    
    def __init__(self):
        # Initialize embeddings model. Offline mode is controlled globally.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.bm25_retriever = None
        self.chunks = []
        
        # Paths for persistence
        self.faiss_path = "data/faiss_index"
        self.bm25_path = "data/bm25_index.pkl"
        self.chunks_path = "data/chunks.pkl"
        self.hash_path = "data/corpus_hash.txt"
        self.lock_path = "data/index.lock"
        self.stale_threshold_sec = 300 # 5 minutes

    def _calculate_corpus_hash(self, directory: str) -> str:
        """Calculate a hash representing the state of all PDF files in the directory."""
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return ""
        pdf_files = sorted(list(path.glob("**/*.pdf")))
        hash_m = hashlib.md5()
        for file in pdf_files:
            try:
                stat = file.stat()
                file_info = f"{file.name}_{stat.st_mtime}_{stat.st_size}"
                hash_m.update(file_info.encode("utf-8"))
            except Exception:
                pass
        return hash_m.hexdigest()

    def create_index(self, documents: List[Document]):
        """Create or load vector store and BM25 index from documents.
        
        Args:
            documents: list of documents to be added to the vector store
        """
        corpus_dir = "data"
        current_hash = self._calculate_corpus_hash(corpus_dir)
        
        # 1. Stale Lock Handling
        lock_file = Path(self.lock_path)
        if lock_file.exists():
            try:
                mtime = lock_file.stat().st_mtime
                age = time.time() - mtime
                if age > self.stale_threshold_sec:
                    print(f"[DEBUG] Found stale lock file '{self.lock_path}' (age: {age:.1f}s). Removing it...", flush=True)
                    lock_file.unlink(missing_ok=True)
            except Exception as e:
                print(f"[DEBUG] Error checking stale lock file: {e}", flush=True)

        # 2. Check if persistent files exist and hash matches
        faiss_dir = Path(self.faiss_path)
        bm25_file = Path(self.bm25_path)
        chunks_file = Path(self.chunks_path)
        hash_file = Path(self.hash_path)
        
        if (faiss_dir.exists() and bm25_file.exists() and chunks_file.exists() and hash_file.exists() and current_hash):
            try:
                stored_hash = hash_file.read_text().strip()
                if stored_hash == current_hash:
                    print("[DEBUG] Corpus hash matches. Loading saved indexes from disk...", flush=True)
                    self.vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
                    with open(self.bm25_path, "rb") as f:
                        self.bm25_retriever = pickle.load(f)
                    with open(self.chunks_path, "rb") as f:
                        self.chunks = pickle.load(f)
                    print(f"[DEBUG] Loaded FAISS and BM25 indexes successfully from disk ({len(self.chunks)} chunks).", flush=True)
                    return
            except Exception as e:
                print(f"[DEBUG] Failed to load cached index: {e}. Rebuilding...", flush=True)

        # 3. Build or load with race condition prevention
        max_attempts = 2
        for attempt in range(max_attempts):
            os.makedirs("data", exist_ok=True)
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                
                # We acquired the lock. Build the indexes!
                print(f"\n[DEBUG] create_index: Embedding {len(documents)} chunks using SentenceTransformer...", flush=True)
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.chunks = documents
                
                print("[DEBUG] create_index: Building BM25 index...", flush=True)
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                
                # Save built indexes and hash to disk
                print("[DEBUG] Saving FAISS, BM25, and chunks indexes to disk...", flush=True)
                self.vectorstore.save_local(self.faiss_path)
                with open(self.bm25_path, "wb") as f:
                    pickle.dump(self.bm25_retriever, f)
                with open(self.chunks_path, "wb") as f:
                    pickle.dump(self.chunks, f)
                if current_hash:
                    hash_file.write_text(current_hash)
                
                # Remove lock file
                try:
                    lock_file.unlink(missing_ok=True)
                except Exception:
                    pass
                print("[DEBUG] FAISS and BM25 indexes built and cached to disk successfully.", flush=True)
                return
                
            except FileExistsError:
                # Another process is building the index. Wait until it's done.
                print(f"[DEBUG] Another process is building the index (attempt {attempt + 1}). Waiting for lock release...", flush=True)
                wait_start = time.time()
                while lock_file.exists():
                    try:
                        mtime = lock_file.stat().st_mtime
                        age = time.time() - mtime
                        if age > self.stale_threshold_sec:
                            print(f"[DEBUG] Lock file became stale while waiting. Removing it...", flush=True)
                            lock_file.unlink(missing_ok=True)
                            break
                    except Exception:
                        pass
                    
                    if time.time() - wait_start > self.stale_threshold_sec:
                        print("[DEBUG] Wait timeout exceeded. Breaking...", flush=True)
                        break
                    time.sleep(1.0)
                
                # Try to load the newly built index from disk
                try:
                    print("[DEBUG] Loading indexes built by the other process...", flush=True)
                    self.vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
                    with open(self.bm25_path, "rb") as f:
                        self.bm25_retriever = pickle.load(f)
                    with open(self.chunks_path, "rb") as f:
                        self.chunks = pickle.load(f)
                    print("[DEBUG] Loaded indexes successfully.", flush=True)
                    return
                except Exception as e:
                    print(f"[DEBUG] Failed to load indexes built by other process: {e}.", flush=True)
                    # If this is our last attempt, raise a RuntimeError
                    if attempt == max_attempts - 1:
                        print("[DEBUG] Rebuilding failed. Forcing lock cleanup and raising RuntimeError.", flush=True)
                        try:
                            lock_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise RuntimeError(f"Failed to load or build RAG index: {e}")

    def retrieve(self, query: str, k: int = 5, rrf_constant: int = 60) -> List[Document]:
        """Perform hybrid search using Reciprocal Rank Fusion (RRF).
        
        Args:
            query: search query
            k: number of top chunks to return
            rrf_constant: constant parameter for RRF formula (default 60)
        Returns:
            List of unique top-k merged Documents
        """
        if self.vectorstore is None or self.bm25_retriever is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        # 1. Vector Search Retrieval (retrieve 2x documents to allow merging)
        vector_docs = self.vectorstore.similarity_search(query, k=k * 2)
        
        # 2. BM25 Search Retrieval (retrieve 2x documents)
        self.bm25_retriever.k = k * 2
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 3. Reciprocal Rank Fusion merging
        rrf_scores: Dict[Tuple[str, int, str], float] = {}
        doc_map: Dict[Tuple[str, int, str], Document] = {}
        
        def add_docs_to_rrf(docs: List[Document]):
            for rank, doc in enumerate(docs, start=1):
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                source = os.path.basename(meta.get("source", "unknown"))
                page = meta.get("page", 0) + 1
                key = (source, page, doc.page_content)
                
                # Accumulate RRF score
                score = 1.0 / (rrf_constant + rank)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + score
                doc_map[key] = doc
        
        add_docs_to_rrf(vector_docs)
        add_docs_to_rrf(bm25_docs)
        
        # 4. Sort and return top-k documents
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_k_docs = [doc_map[key] for key in sorted_keys[:k]]
        
        return top_k_docs
