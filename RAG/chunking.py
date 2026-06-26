import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

# Setup router
router = APIRouter()

# Setup logger
logger = logging.getLogger(__name__)


class ChunkRequest(BaseModel):
    """Request model for document chunking"""
    content: str                        # Raw text content to chunk
    chunk_size: Optional[int] = 1000    # Max characters per chunk
    chunk_overlap: Optional[int] = 200  # Overlap between chunks
    source: Optional[str] = None        # Optional source label (URL or file path)


class ChunkResponse(BaseModel):
    """Response model for chunked documents"""
    status: str
    source: Optional[str]
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    total_characters: int
    chunks: list[dict]


@router.post("", response_model=ChunkResponse)
async def chunk_document(request: ChunkRequest):
    """
    Accepts raw text content and splits it into overlapping chunks
    using LangChain's RecursiveCharacterTextSplitter.

    - chunk_size:    maximum number of characters per chunk (default 1000)
    - chunk_overlap: number of characters shared between adjacent chunks (default 200)
    """
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    if request.chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be a positive integer")

    if request.chunk_overlap < 0:
        raise HTTPException(status_code=400, detail="chunk_overlap cannot be negative")

    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size"
        )

    try:
        logger.info(
            f"Chunking content — size={request.chunk_size}, "
            f"overlap={request.chunk_overlap}, source={request.source!r}"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        raw_chunks = splitter.split_text(request.content)

        chunks = [
            {
                "chunk_index": idx,
                "character_count": len(text),
                "text": text,
            }
            for idx, text in enumerate(raw_chunks)
        ]

        logger.info(f"Produced {len(chunks)} chunks from {len(request.content)} characters")

        return ChunkResponse(
            status="success",
            source=request.source,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            total_chunks=len(chunks),
            total_characters=len(request.content),
            chunks=chunks,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chunking failed")
        raise HTTPException(status_code=500, detail=str(e))