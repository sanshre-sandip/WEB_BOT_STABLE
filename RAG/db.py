import logging
import uuid
from collections import Counter
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .vectorstore import get_vectorstore

router = APIRouter()
logger = logging.getLogger(__name__)

# ----------------------------------
# Request / Response Models
# ----------------------------------
class StoreRequest(BaseModel):
    chunks: list[str]               # Text chunks to store
    source: Optional[str] = None    # Optional source label (URL or file path)


class StoreResponse(BaseModel):
    status: str
    source: Optional[str]
    chunks_stored: int
    total_in_db: int
    ids: list[str]


class SourceRequest(BaseModel):
    source: str


# ----------------------------------
# Endpoints
# ----------------------------------

@router.post("/store", response_model=StoreResponse)
async def store_chunks(request: StoreRequest):
    """
    Store text chunks into Chroma vector store.
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty")

    valid_chunks = [c for c in request.chunks if c.strip()]
    if not valid_chunks:
        raise HTTPException(status_code=400, detail="All chunks are empty or whitespace")

    try:
        store = get_vectorstore()
        ids = [str(uuid.uuid4()) for _ in valid_chunks]
        metadatas = [{"source": request.source or "unknown"} for _ in valid_chunks]

        store.add_texts(
            texts=valid_chunks,
            metadatas=metadatas,
            ids=ids,
        )

        # Public count workaround: get all IDs
        total = len(store.get(include=[])["ids"])
        logger.info(f"Stored {len(valid_chunks)} chunks — total in DB: {total}")

        return StoreResponse(
            status="success",
            source=request.source,
            chunks_stored=len(valid_chunks),
            total_in_db=total,
            ids=ids,
        )

    except Exception as e:
        logger.exception("Store failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def db_info():
    """
    Database overview.
    """
    try:
        store = get_vectorstore()
        result = store.get(include=["metadatas"])

        total_docs = len(result["ids"])
        sources = [meta.get("source", "unknown") for meta in result["metadatas"]]
        source_counts = dict(Counter(sources))

        return {
            "total_documents": total_docs,
            "unique_sources": len(source_counts),
            "sources": source_counts,
            "persist_directory": "./chroma_db",
            "embedding_model": "all-MiniLM-L6-v2"
        }

    except Exception as e:
        logger.exception("DB info failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources")
async def list_sources():
    """
    List all sources and chunk counts.
    """
    try:
        store = get_vectorstore()
        result = store.get(include=["metadatas"])

        sources = [meta.get("source", "unknown") for meta in result["metadatas"]]
        counts = Counter(sources)

        output = [
            {"source": source, "chunk_count": count}
            for source, count in counts.items()
        ]
        output.sort(key=lambda x: x["chunk_count"], reverse=True)

        return {
            "total_sources": len(output),
            "sources": output
        }

    except Exception as e:
        logger.exception("List sources failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_chunks(
    source: str | None = None,
    limit: int = 20,
    offset: int = 0
):
    """
    Paginated chunk listing.
    """
    try:
        store = get_vectorstore()
        # Filter by source if provided using public 'where'
        where = {"source": source} if source else None
        
        result = store.get(
            where=where,
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset
        )

        chunks = []
        for chunk_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            chunks.append({
                "id": chunk_id,
                "source": meta.get("source", "unknown"),
                "char_count": len(doc),
                "preview": doc[:200]
            })

        # To get total_matching without pagination, we might need another get()
        # but for simplicity we'll just return what we have or a total count
        total_matching = len(store.get(where=where, include=[])["ids"])

        return {
            "source_filter": source,
            "total_matching": total_matching,
            "returned": len(chunks),
            "limit": limit,
            "offset": offset,
            "chunks": chunks
        }

    except Exception as e:
        logger.exception("List chunks failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{chunk_id}")
async def get_chunk(chunk_id: str):
    """
    Fetch one chunk by ID.
    """
    try:
        store = get_vectorstore()
        result = store.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )

        if not result["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"Chunk not found: {chunk_id}"
            )

        return {
            "status": "success",
            "id": result["ids"][0],
            "source": result["metadatas"][0].get("source", "unknown"),
            "text": result["documents"][0]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Get chunk failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-source")
async def clear_source(request: SourceRequest):
    """
    Delete all chunks from a source.
    """
    try:
        source = request.source.strip()
        if not source:
            raise HTTPException(status_code=400, detail="Source cannot be empty")

        store = get_vectorstore()
        result = store.get(where={"source": source}, include=[])
        ids_to_delete = result.get("ids", [])

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: {source}"
            )

        store.delete(ids=ids_to_delete)
        remaining = len(store.get(include=[])["ids"])

        return {
            "message": f"Removed source '{source}'",
            "chunks_deleted": len(ids_to_delete),
            "total_remaining": remaining
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Clear source failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear-all")
async def clear_all():
    """
    Delete entire database.
    """
    try:
        store = get_vectorstore()
        result = store.get(include=[])
        count_before = len(result["ids"])

        if result["ids"]:
            store.delete(ids=result["ids"])

        return {
            "message": "Database cleared",
            "chunks_deleted": count_before,
            "total_remaining": 0
        }

    except Exception as e:
        logger.exception("Clear all failed")
        raise HTTPException(status_code=500, detail=str(e))