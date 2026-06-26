import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx
import streamlit as st
from dotenv import load_dotenv

from RAG.config import (
    BACKEND_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_VECTOR_SOURCE_LABEL,
)

load_dotenv()

st.set_page_config(page_title="Document Loading Pipeline", layout="wide")

DEFAULT_WEB_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"
DEFAULT_PDF_PATH = ""
DEFAULT_TIMEOUT = 240.0


def get_backend_url() -> str:
    return os.getenv("BACKEND", BACKEND_URL).rstrip("/")


def make_system_source_label(source: str) -> str:
    raw_name = source.strip().split("/")[-1].strip() or "document"
    safe_name = "".join(
        char if char.isalnum() or char in "-_." else "-"
        for char in raw_name
    ).strip("-.")[:80]
    safe_name = safe_name or "document"
    return f"{DEFAULT_VECTOR_SOURCE_LABEL}-{safe_name}-{int(time.time())}"


def call_backend(
    method: str,
    path: str,
    backend_url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    url = f"{backend_url.rstrip('/')}{path}"
    kwargs: dict[str, Any] = {"timeout": timeout}
    if payload is not None:
        kwargs["json"] = payload

    try:
        response = httpx.request(method, url, **kwargs)
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to backend at {backend_url}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out: {method} {path}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"HTTP request failed: {method} {path}: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} returned {response.status_code}: {response.text[:500]}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Response from {path} was not valid JSON: {response.text[:500]}") from exc


def initialize_session_state() -> None:
    st.session_state.setdefault("backend_url", get_backend_url())
    st.session_state.setdefault("loaded_content", "")
    st.session_state.setdefault("load_result", None)
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("chunk_result", None)
    st.session_state.setdefault("embed_result", None)
    st.session_state.setdefault("store_result", None)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("current_source_label", DEFAULT_VECTOR_SOURCE_LABEL)
    st.session_state.setdefault("current_source", "")


def clear_pipeline_state() -> None:
    st.session_state.loaded_content = ""
    st.session_state.load_result = None
    st.session_state.chunks = []
    st.session_state.chunk_result = None
    st.session_state.embed_result = None
    st.session_state.store_result = None
    st.session_state.pipeline_result = None
    st.session_state.current_source_label = DEFAULT_VECTOR_SOURCE_LABEL
    st.session_state.current_source = ""


def load_document(backend_url: str, source_type: str, source: str, source_label: str) -> dict[str, Any]:
    result = call_backend(
        "POST",
        "/process",
        backend_url,
        payload={"type": source_type, "source": source},
    )
    content = result.get("content") or result.get("preview", "")
    if not content:
        raise RuntimeError("Document loader returned no content")

    st.session_state.loaded_content = content
    st.session_state.load_result = result
    st.session_state.current_source_label = source_label
    st.session_state.current_source = source
    return result


def chunk_loaded_content(backend_url: str) -> dict[str, Any]:
    content = st.session_state.loaded_content
    if not content:
        raise RuntimeError("Load a document before chunking")

    result = call_backend(
        "POST",
        "/chunk",
        backend_url,
        payload={
            "content": content,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "source": st.session_state.current_source_label,
        },
    )
    st.session_state.chunks = [chunk["text"] for chunk in result.get("chunks", [])]
    st.session_state.chunk_result = result
    return result


def embed_chunks(backend_url: str) -> dict[str, Any]:
    chunks = st.session_state.chunks
    if not chunks:
        raise RuntimeError("Create chunks before embedding")

    result = call_backend(
        "POST",
        "/embed",
        backend_url,
        payload={"chunks": chunks, "source": st.session_state.current_source_label},
    )
    st.session_state.embed_result = result
    return result


def store_chunks(backend_url: str) -> dict[str, Any]:
    chunks = st.session_state.chunks
    if not chunks:
        raise RuntimeError("Create chunks before storing")

    result = call_backend(
        "POST",
        "/vectorstore/store",
        backend_url,
        payload={"chunks": chunks, "source": st.session_state.current_source_label},
    )
    st.session_state.store_result = result
    return result


def run_full_pipeline(
    backend_url: str,
    source_type: str,
    source: str,
) -> dict[str, Any]:
    progress = st.progress(0)
    source_label = make_system_source_label(source)

    load_result = load_document(backend_url, source_type, source, source_label)
    progress.progress(25)

    chunk_result = chunk_loaded_content(backend_url)
    progress.progress(50)

    embed_result = embed_chunks(backend_url)
    progress.progress(75)

    store_result = store_chunks(backend_url)
    progress.progress(100)

    result = {
        "load": load_result,
        "chunk": chunk_result,
        "embed": embed_result,
        "store": store_result,
        "source_label": source_label,
    }
    st.session_state.pipeline_result = result
    return result


def show_metrics() -> None:
    cols = st.columns(5)

    with cols[0]:
        st.metric("Loaded Characters", len(st.session_state.loaded_content))

    with cols[1]:
        st.metric("Chunks", len(st.session_state.chunks))

    embedding_dim = None
    if st.session_state.embed_result:
        embedding_dim = st.session_state.embed_result.get("embedding_dim")
    with cols[2]:
        st.metric("Embedding Dim", embedding_dim or "Not set")

    chunks_stored = None
    if st.session_state.store_result:
        chunks_stored = st.session_state.store_result.get("chunks_stored")
    with cols[3]:
        st.metric("Chunks Stored", chunks_stored or "Not set")

    with cols[4]:
        st.metric("System Chunk Size", DEFAULT_CHUNK_SIZE)


def show_load_result() -> None:
    result = st.session_state.load_result
    if not result:
        return

    st.subheader("1. Document Loading")
    st.json(result)
    preview = result.get("preview", "")
    if preview:
        with st.expander("Loaded content preview"):
            st.code(preview)


def show_chunk_result() -> None:
    result = st.session_state.chunk_result
    chunks = st.session_state.chunks
    if not result or not chunks:
        return

    st.subheader("2. Chunking")
    st.json(result)

    rows = [
        {
            "Index": chunk.get("chunk_index"),
            "Characters": chunk.get("character_count"),
            "Preview": chunk.get("text", "")[:200],
        }
        for chunk in result.get("chunks", [])
    ]
    st.dataframe(rows, use_container_width=True)


def show_embed_result() -> None:
    result = st.session_state.embed_result
    if not result:
        return

    st.subheader("3. Embedding")
    st.json(result)


def show_store_result() -> None:
    result = st.session_state.store_result
    if not result:
        return

    st.subheader("4. Vector Store")
    st.json(result)


def render_pipeline_result() -> None:
    result = st.session_state.pipeline_result
    if not result:
        return

    st.subheader("Full Pipeline Result")
    st.json({
        "source_label": result.get("source_label"),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "load": {
            "document_count": result["load"].get("document_count"),
            "processed_length": result["load"].get("processed_length"),
            "source_processed": result["load"].get("source_processed"),
        },
        "chunk": {
            "total_chunks": result["chunk"].get("total_chunks"),
            "chunk_size": result["chunk"].get("chunk_size"),
            "chunk_overlap": result["chunk"].get("chunk_overlap"),
        },
        "embed": {
            "total_chunks": result["embed"].get("total_chunks"),
            "embedding_dim": result["embed"].get("embedding_dim"),
            "model": result["embed"].get("model"),
        },
        "store": {
            "chunks_stored": result["store"].get("chunks_stored"),
            "total_in_db": result["store"].get("total_in_db"),
            "source": result["store"].get("source"),
        },
    })


def main() -> None:
    initialize_session_state()

    st.title("Document Loading Pipeline")
    st.caption("Load a document, chunk it, embed it, and store it in the vector database using system defaults.")

    st.sidebar.title("System Pipeline Settings")
    backend_url = st.sidebar.text_input("Backend URL", value=st.session_state.backend_url)
    st.session_state.backend_url = backend_url.rstrip("/")

    source_type = st.sidebar.radio("Source Type", ["web", "pdf"], index=0)
    default_source = DEFAULT_WEB_URL if source_type == "web" else DEFAULT_PDF_PATH
    source = st.sidebar.text_input("Web URL or PDF Path", value=default_source)

    st.sidebar.divider()
    st.sidebar.caption(f"Vector DB source label: generated by system")
    st.sidebar.caption(f"Chunk size: {DEFAULT_CHUNK_SIZE}")
    st.sidebar.caption(f"Chunk overlap: {DEFAULT_CHUNK_OVERLAP}")
    st.sidebar.caption(f"Default source prefix: {DEFAULT_VECTOR_SOURCE_LABEL}")

    st.sidebar.divider()

    if st.sidebar.button("Load"):
        try:
            result = load_document(
                backend_url.rstrip("/"),
                source_type,
                source,
                make_system_source_label(source),
            )
            st.success(f"Loaded {result.get('document_count')} document(s)")
        except Exception as exc:
            st.error(str(exc))

    if st.sidebar.button("Chunk"):
        try:
            result = chunk_loaded_content(backend_url.rstrip("/"))
            st.success(f"Created {result.get('total_chunks')} chunks")
        except Exception as exc:
            st.error(str(exc))

    if st.sidebar.button("Clear"):
        clear_pipeline_state()
        st.success("Pipeline state cleared")

    if st.sidebar.button("Embed Chunks"):
        try:
            result = embed_chunks(backend_url.rstrip("/"))
            st.success(f"Embedded {result.get('total_chunks')} chunks with dim {result.get('embedding_dim')}")
        except Exception as exc:
            st.error(str(exc))

    if st.sidebar.button("Store Chunks"):
        try:
            result = store_chunks(backend_url.rstrip("/"))
            st.success(f"Stored {result.get('chunks_stored')} chunks. Total in DB: {result.get('total_in_db')}")
        except Exception as exc:
            st.error(str(exc))

    if st.sidebar.button("Run Full Pipeline", type="primary"):
        try:
            result = run_full_pipeline(
                backend_url.rstrip("/"),
                source_type,
                source,
            )
            st.success(
                "Pipeline complete: "
                f"loaded={result['load'].get('document_count')}, "
                f"chunks={result['chunk'].get('total_chunks')}, "
                f"stored={result['store'].get('chunks_stored')}"
            )
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    show_metrics()
    render_pipeline_result()

    st.divider()
    show_load_result()
    show_chunk_result()
    show_embed_result()
    show_store_result()


if __name__ == "__main__":
    main()
