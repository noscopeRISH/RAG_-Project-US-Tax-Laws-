from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    source: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class HealthResponse(BaseModel):
    status: str
    chunks_count: int
