import os
import logging
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# Setup router
router = APIRouter()

# Setup logger
logger = logging.getLogger(__name__)

class SourceRequest(BaseModel):
    """Request model for document loading"""
    type: str  # "pdf" or "web"
    source: str  # file path or URL

def is_valid_url(url: str) -> bool:
    """Validate if a string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

@router.post("")
async def process_source(request: SourceRequest):
    """
    Endpoint logic moved to router.
    Note: The prefix '/process' is handled in main.py
    """
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="Source cannot be empty")

    try:
        if request.type.lower() == "pdf":
            if not os.path.exists(request.source):
                raise HTTPException(status_code=404, detail=f"PDF not found: {request.source}")
            logger.info(f"Loading PDF: {request.source}")
            loader = PyPDFLoader(request.source)
            docs = loader.load()

        elif request.type.lower() == "web":
            if not is_valid_url(request.source):
                raise HTTPException(status_code=400, detail="Invalid URL")
            logger.info(f"Loading Website: {request.source}")
            loader = WebBaseLoader(request.source)
            docs = loader.load()

        else:
            raise HTTPException(status_code=400, detail="Type must be 'web' or 'pdf'")

        content = "\n".join([doc.page_content for doc in docs])
        logger.info(f"Loaded {len(docs)} documents successfully")

        return {
            "status": "success",
            "loader_used": request.type,
            "source_processed": request.source,
            "document_count": len(docs),
            "processed_length": len(content),
            "content": content,
            "preview": content[:1000],
            "message": "Document loaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing Failed")
        raise HTTPException(status_code=500, detail=str(e))
