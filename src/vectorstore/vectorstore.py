"""vector store module for document embedding and retrieval"""

import os
import pickle
import hashlib
import time
from typing import List, Union
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever


class VectorStore:
    """Manages vector store applications and hybrid search retrieval"""
    def __init__(self):
        # Initialize embeddings model. Note: Offline mode is controlled globally by 
        # setting os.environ["HF_HUB_OFFLINE"] = "1" at the app entrypoints (config.py/streamlit_app.py).
        # This forces the library to load the model locally from the Hugging Face cache.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.retriever = None
        self.bm25_retriever = None
        
        # Paths for persistence
        self.faiss_path = "data/faiss_index"
        self.bm25_path = "data/bm25_index.pkl"
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

    def create_vectorstore(self, documents: List[Document]):
        """Create or load vector store and BM25 index from documents.
        Args:
            documents: list of documents to be added to the vector store
        """
        # Determine corpus directory
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
        hash_file = Path(self.hash_path)
        
        if (faiss_dir.exists() and bm25_file.exists() and hash_file.exists() and current_hash):
            try:
                stored_hash = hash_file.read_text().strip()
                if stored_hash == current_hash:
                    print("[DEBUG] Corpus hash matches. Loading saved indexes from disk...", flush=True)
                    self.vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
                    self.retriever = self.vectorstore.as_retriever()
                    with open(self.bm25_path, "rb") as f:
                        self.bm25_retriever = pickle.load(f)
                    print("[DEBUG] Loaded FAISS and BM25 indexes successfully from disk.", flush=True)
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
                
                # We acquired the lock. Build the index!
                print(f"\n[DEBUG] create_vectorstore: Embedding {len(documents)} chunks using SentenceTransformer...", flush=True)
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.retriever = self.vectorstore.as_retriever()
                
                print("[DEBUG] create_vectorstore: Building BM25 index...", flush=True)
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = 4
                
                # Save built indexes and hash to disk
                print("[DEBUG] Saving FAISS and BM25 indexes to disk...", flush=True)
                self.vectorstore.save_local(self.faiss_path)
                with open(self.bm25_path, "wb") as f:
                    pickle.dump(self.bm25_retriever, f)
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
                    self.retriever = self.vectorstore.as_retriever()
                    with open(self.bm25_path, "rb") as f:
                        self.bm25_retriever = pickle.load(f)
                    print("[DEBUG] Loaded indexes successfully.", flush=True)
                    return
                except Exception as e:
                    print(f"[DEBUG] Failed to load indexes built by other process: {e}.", flush=True)
                    # If this is our last attempt, we raise a RuntimeError
                    if attempt == max_attempts - 1:
                        print("[DEBUG] Rebuilding failed. Forcing lock cleanup and raising RuntimeError.", flush=True)
                        try:
                            lock_file.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise RuntimeError(f"Failed to load or build RAG index: {e}")

    def create_retriever(self, documents: List[Document]):
        """Backward-compatible alias for create_vectorstore."""
        self.create_vectorstore(documents)
    
    def get_retriever(self):
        """Get the retriever instance
        Returns:
            retriever instance 
        """
        if self.retriever is None:
            raise ValueError("vector store not initialized. call create_vectorstore first.")
        return self.retriever

    def get_hybrid_retriever(self, keyword_weight: float = 0.4, vector_weight: float = 0.6) -> EnsembleRetriever:
        """Get the hybrid EnsembleRetriever combining BM25 and FAISS vector retriever.
        
        Args:
            keyword_weight (float): BM25 retrieval weight.
            vector_weight (float): FAISS vector retrieval weight.
        Returns:
            EnsembleRetriever: The combined retriever instance.
        """
        if self.retriever is None or self.bm25_retriever is None:
            raise ValueError("vector store not initialized. call create_vectorstore first.")
        
        # Ensure BM25 retriever returns k results
        self.bm25_retriever.k = 4
        
        # Note on ensemble weights alignment:
        # retrievers[0] (bm25_retriever) maps to weights[0] (keyword_weight)
        # retrievers[1] (self.retriever vector retriever) maps to weights[1] (vector_weight)
        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.retriever],
            weights=[keyword_weight, vector_weight]
        )
    
    def retrieve(self, query: str, k: int = 4)-> List[Document]:
        """Retrieve documents for a query

        Args:
            query: search query
            k: number of documents to retrieve
        Returns:
            list of retrieved documents
        """
        if self.retriever is None:
            raise ValueError("vector store not initialized. call create_vectorstore first.")
        return self.vectorstore.similarity_search(query, k=k)

    def retireve(self, query: str, k: int = 4)-> List[Document]:
        """Backward-compatible alias for retrieve."""
        return self.retrieve(query, k=k)
