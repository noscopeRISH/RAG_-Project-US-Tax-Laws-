"""document processing module for loading and splitting documents"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.documents  import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import(
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)


class DocumentProcessor:
    """handles document loading and processing"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):

        """
        Initializes document processor 
        
        Args:
            chunk_size (int): The size text chunk.
            chunk_overlap (int): The overlap between chunks.
        """
    
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
    
    def load_from_url(self,url:str)->List[Document]:
        """load documents from urls"""
        loader = WebBaseLoader(url)
        return loader.load()
       
    def load_from_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory individually with logging."""
        path = Path(directory)
        docs = []
        # Find all PDFs recursively
        pdf_files = sorted(list(path.glob("**/*.pdf")))
        print(f"\n[DEBUG] Ingestion: Found {len(pdf_files)} PDF files in '{directory}'")
        for i, file in enumerate(pdf_files, 1):
            print(f"[DEBUG] [{i}/{len(pdf_files)}] Loading PDF: {file.name}...", flush=True)
            try:
                loader = PyPDFLoader(str(file))
                loaded = loader.load()
                docs.extend(loaded)
                print(f"[DEBUG] [{i}/{len(pdf_files)}] Success: loaded {len(loaded)} pages.", flush=True)
            except Exception as e:
                print(f"[DEBUG] [{i}/{len(pdf_files)}] FAILED to load {file.name}: {e}", flush=True)
        return docs
    
    def load_from_pdf(self, file_path: Union[str, Path])-> List[Document]:
        """ Load Documents from a pdf file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()        
    
    def load_from_txt(self,file_path: Union[str, Path])-> List[Document]:
        """Load documents from a txt file"""
        loader = TextLoader(str(file_path), encoding= "utf-8")
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or text files
        Args:
            sources: list of Pdf folder paths , or URLs
        Returns:
            list of loaded documents
        """
    
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue
            path = Path(src)
            if path.is_dir():
                docs.extend(self.load_from_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            else: 
                raise ValueError(
                    f"Unsupported source type: {src}."
                    " Please provide a valid URL, PDF directory, PDF file, or text file."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) ->List[Document]:
        """
        Split documents into smaller chunks using the specified chunk size and overlap.
        Args:
            documents: list of loaded documents
        Returns:
            list of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        complete pipeline to load and split documents 
        
        Args:
            sources: list of Pdf folder paths , or URLs
        Returns:
            list of processed document chunks"""
        
        print("\n[DEBUG] process_urls: Starting document loading...", flush=True)
        docs = self.load_documents(urls)
        print(f"[DEBUG] process_urls: Loaded {len(docs)} pages in total. Splitting into chunks...", flush=True)
        split_docs = self.split_documents(docs)
        print(f"[DEBUG] process_urls: Successfully generated {len(split_docs)} document chunks.", flush=True)
        return split_docs
