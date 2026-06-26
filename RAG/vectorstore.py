import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from langchain_chroma import Chroma
from pydantic import BaseModel

from .config import CHROMA_DIR
from .embedding import get_embedder

router = APIRouter()
logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embedder = get_embedder()
        logger.info(f"Connecting to Chroma DB — persist dir: {CHROMA_DIR}")
        _vectorstore = Chroma(
            collection_name="web_bot",
            embedding_function=embedder,
            persist_directory=CHROMA_DIR,
        )
    return _vectorstore


class StoreRequest(BaseModel):
    chunks: list[str]
    source: Optional[str] = None


class StoreResponse(BaseModel):
    status: str
    source: Optional[str]
    chunks_stored: int
    total_in_db: int
    ids: list[str]


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    rank: int
    score: float
    text: str
    source: Optional[str]


class SearchResponse(BaseModel):
    status: str
    query: str
    top_k: int
    results: list[SearchResult]


class DeleteRequest(BaseModel):
    source: str


class DeleteResponse(BaseModel):
    status: str
    source: str
    chunks_deleted: int
    total_in_db: int


class StatsResponse(BaseModel):
    status: str
    total_documents: int
    persist_directory: str
    embedding_model: str


def count_documents(store: Chroma) -> int:
    return len(store.get(include=[])["ids"])


def validate_chunks(chunks: list[str]) -> list[str]:
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty")

    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    if not valid_chunks:
        raise HTTPException(status_code=400, detail="All chunks are empty or whitespace")

    return valid_chunks


@router.post("/store", response_model=StoreResponse)
async def store_chunks(request: StoreRequest):
    try:
        valid_chunks = validate_chunks(request.chunks)
        store = get_vectorstore()
        ids = [str(uuid.uuid4()) for _ in valid_chunks]
        metadatas = [
            {"source": request.source or "unknown", "id": chunk_id}
            for chunk_id in ids
        ]

        store.add_texts(
            texts=valid_chunks,
            metadatas=metadatas,
            ids=ids,
        )

        total = count_documents(store)
        logger.info(f"Stored {len(valid_chunks)} chunks — total in DB: {total}")

        return StoreResponse(
            status="success",
            source=request.source,
            chunks_stored=len(valid_chunks),
            total_in_db=total,
            ids=ids,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Store failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer")

    try:
        store = get_vectorstore()
        existing = store.get(limit=1)
        if not existing["ids"]:
            raise HTTPException(status_code=404, detail="Vector store is empty — store some chunks first")

        results = store.similarity_search_with_relevance_scores(
            query=request.query,
            k=min(request.top_k, count_documents(store)),
        )

        search_results = [
            SearchResult(
                rank=idx + 1,
                score=round(score, 4),
                text=doc.page_content,
                source=doc.metadata.get("source"),
            )
            for idx, (doc, score) in enumerate(results)
        ]

        logger.info(f"Search query={request.query!r} — returned {len(search_results)} results")

        return SearchResponse(
            status="success",
            query=request.query,
            top_k=request.top_k,
            results=search_results,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/delete", response_model=DeleteResponse)
async def delete_by_source(request: DeleteRequest):
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="Source cannot be empty")

    try:
        store = get_vectorstore()
        existing = store.get(where={"source": request.source})
        ids_to_delete = existing.get("ids", [])

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source: {request.source!r}",
            )

        store.delete(ids=ids_to_delete)
        total = count_documents(store)

        logger.info(f"Deleted {len(ids_to_delete)} chunks for source={request.source!r}")

        return DeleteResponse(
            status="success",
            source=request.source,
            chunks_deleted=len(ids_to_delete),
            total_in_db=total,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    try:
        store = get_vectorstore()
        return StatsResponse(
            status="success",
            total_documents=count_documents(store),
            persist_directory=CHROMA_DIR,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    except Exception as exc:
        logger.exception("Stats failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
