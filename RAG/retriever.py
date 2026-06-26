import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from .vectorstore import get_vectorstore

# ----------------------------------
# Logging
# ----------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------
# Router
# ----------------------------------
router = APIRouter()


# ----------------------------------
# Schemas
# ----------------------------------
class QueryRequest(BaseModel):
    query: str
    source: Optional[str] = None   # filter by source (optional)
    k: int = 5                     # number of results to return


# ----------------------------------
# Endpoints
# ----------------------------------

@router.post("/query")
async def query_chunks(req: QueryRequest):
    """
    Semantic search — returns top-k most relevant chunks for a query.
    Optionally filter by source.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if req.k < 1 or req.k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    try:
        store = get_vectorstore()
        filter_dict = {"source": req.source} if req.source else None

        results = store.similarity_search_with_score(
            query=req.query,
            k=req.k,
            filter=filter_dict,
        )

        chunks = [
            {
                "id": doc.metadata.get("id", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "text": doc.page_content,
                "score": round(float(score), 6),
            }
            for doc, score in results
        ]

        return {
            "status": "success",
            "query": req.query,
            "source_filter": req.source,
            "k": req.k,
            "count": len(chunks),
            "results": chunks,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-multi-source")
async def query_multi_source(req: QueryRequest):
    """
    Run the same query across all sources separately.
    Returns top-k results per source.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        store = get_vectorstore()
        # Public replacement for _collection.get()
        raw = store.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in raw["metadatas"]})

        if not sources:
            return {
                "status": "success",
                "query": req.query,
                "results_by_source": {},
            }

        results_by_source = {}
        for source in sorted(sources):
            hits = store.similarity_search_with_score(
                query=req.query,
                k=req.k,
                filter={"source": source},
            )
            results_by_source[source] = [
                {
                    "id": doc.metadata.get("id", "unknown"),
                    "text": doc.page_content,
                    "score": round(float(score), 6),
                }
                for doc, score in hits
            ]

        return {
            "status": "success",
            "query": req.query,
            "k_per_source": req.k,
            "sources_searched": len(sources),
            "results_by_source": results_by_source,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Multi-source query failed")
        raise HTTPException(status_code=500, detail=str(e))