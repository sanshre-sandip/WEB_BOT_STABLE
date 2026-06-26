import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

# ----------------------------------
# Setup
# ----------------------------------
router = APIRouter()
logger = logging.getLogger(__name__)

# Free open-source model from HuggingFace (no API key needed)
# all-MiniLM-L6-v2: fast, lightweight, great for semantic search
# EMBEDDING_MODEL is now imported from .config

_embedder = None
...
def get_embedder():
    """
    Lazy-load the embedding model.
    """
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _embedder


# ----------------------------------
# Request / Response Models
# ----------------------------------
class EmbedRequest(BaseModel):
    chunks: list[str]               # List of text chunks to embed
    source: Optional[str] = None    # Optional label for tracking


class EmbedResponse(BaseModel):
    status: str
    source: Optional[str]
    model: str
    total_chunks: int
    embedding_dim: int
    embeddings: list[dict]          # [{chunk_index, text_preview, embedding}]


# ----------------------------------
# Endpoint
# ----------------------------------
@router.post("", response_model=EmbedResponse)
async def embed_chunks(request: EmbedRequest):
    """
    Accepts a list of text chunks and returns a vector embedding for each.

    Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2):
    - Free, no API key required
    - Runs locally on CPU
    - 384-dimensional embeddings
    - Great for semantic search and RAG pipelines
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty")

    # Filter out blank chunks
    valid_chunks = [c for c in request.chunks if c.strip()]
    if not valid_chunks:
        raise HTTPException(status_code=400, detail="All chunks are empty or whitespace")

    try:
        logger.info(f"Embedding {len(valid_chunks)} chunks from source={request.source!r}")

        embedder = get_embedder()
        vectors = embedder.embed_documents(valid_chunks)


        embeddings = [
            {
                "chunk_index": idx,
                "text_preview": chunk[:100].strip(),
                "embedding": vector,
            }
            for idx, (chunk, vector) in enumerate(zip(valid_chunks, vectors))
        ]

        embedding_dim = len(vectors[0]) if vectors else 0
        logger.info(f"Embedded {len(embeddings)} chunks, dim={embedding_dim}")

        return EmbedResponse(
            status="success",
            source=request.source,
            model=EMBEDDING_MODEL,
            total_chunks=len(embeddings),
            embedding_dim=embedding_dim,
            embeddings=embeddings,
        )

    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))