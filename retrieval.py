import os
import warnings
import logging

# Suppress HuggingFace / Transformers warnings and progress bars
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()
try:
    transformers.logging.disable_progress_bar()
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
from schemas import Chunk
from qdrant_client_factory import get_qdrant_client

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def retrieve(query: str, category: str | None = None, top_k: int = TOP_K) -> list[Chunk]:
    """Embed query, search Qdrant, return top-k chunks with scores."""
    embedder = get_embedder()
    qdrant = get_qdrant_client()
    
    query_vector = embedder.encode(query, show_progress_bar=False).tolist()
    
    query_filter = None
    if category:
        query_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])
    
    # Use query_points (qdrant-client >= 1.12)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k
    )
    
    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunk = Chunk(
            id=payload.get("id", str(point.id)),
            text=payload.get("text", ""),
            source_file=payload.get("source_file", ""),
            category=payload.get("category", ""),
            section_title=payload.get("section_title", ""),
            char_start=payload.get("char_start", 0),
            char_end=payload.get("char_end", 0),
            page_number=payload.get("page_number"),
            split_method=payload.get("split_method", "structural"),
            score=point.score
        )
        chunks.append(chunk)
        
    return chunks
