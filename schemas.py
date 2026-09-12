from pydantic import BaseModel
from typing import Literal

class Chunk(BaseModel):
    id: str | int
    text: str
    source_file: str
    category: str
    section_title: str
    char_start: int
    char_end: int
    page_number: int | None = None
    split_method: str  # 'structural' or 'recursive'
    score: float = 0.0  # similarity score from retrieval

class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section_title: str

class Response(BaseModel):
    state: Literal["ANSWERED", "NOT_IN_CORPUS", "CONTRADICTION"]
    answer: str
    citations: list[Citation]
    conflicting_chunks: list[list[str]]  # pairs of chunk_ids
    confidence: float  # 0.0-1.0
    reasoning_note: str
