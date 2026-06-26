"""Shared HTTP client for the FastAPI RAG/generate endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

from RAG.config import (  # noqa: E402
    BACKEND_URL,
    CLIENT_REQUEST_TIMEOUT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_RAG_K,
    DEFAULT_RAG_SOURCE_FILTER,
    DEFAULT_USE_RAG,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_REQUEST_TIMEOUT,
)


def get_backend_url() -> str:
    return os.getenv("BACKEND", BACKEND_URL).rstrip("/")


def _resolve_model(provider: str | None, model: str | None) -> str | None:
    if not model or not model.strip():
        return None
    normalized = model.strip()
    if provider and normalized.lower() == provider.lower():
        return None
    return normalized


def build_rag_payload(
    query: str,
    history: list[dict[str, str]],
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    use_rag: bool | None = None,
    source: str | None = None,
    k: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the same payload Streamlit chat sends to the backend."""
    provider = provider if provider is not None else LLM_PROVIDER
    model = _resolve_model(provider, model if model is not None else DEFAULT_LLM_MODEL)
    temperature = DEFAULT_LLM_TEMPERATURE if temperature is None else temperature
    use_rag = DEFAULT_USE_RAG if use_rag is None else use_rag
    source = DEFAULT_RAG_SOURCE_FILTER if source is None else source
    k = DEFAULT_RAG_K if k is None else k

    if use_rag:
        payload: dict[str, Any] = {
            "query": query,
            "source": source or None,
            "k": k,
            "history": history,
            "temperature": temperature,
            "stream": False,
        }
        if provider:
            payload["provider"] = provider
        if model:
            payload["model"] = model
        return "/generate/rag", payload

    payload = {
        "query": query,
        "context": ["No retrieval context provided."],
        "history": history,
        "temperature": temperature,
        "stream": False,
    }
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    return "/generate", payload


def _local_rag_answer(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    source: str | None = None,
    k: int | None = None,
) -> str:
    """Run RAG generation in-process when the backend is the local app."""
    from llm.generator import Message, construct_prompt, generate_with_fallback
    from RAG.vectorstore import get_vectorstore

    provider = provider if provider is not None else LLM_PROVIDER
    model = _resolve_model(provider, model if model is not None else DEFAULT_LLM_MODEL)
    temperature = DEFAULT_LLM_TEMPERATURE if temperature is None else temperature
    k = DEFAULT_RAG_K if k is None else k

    store = get_vectorstore()
    if store is None:
        raise RuntimeError("Vector store not initialized or unavailable")

    filter_dict = {"source": source} if source else None
    docs = store.similarity_search(query, k=k, filter=filter_dict)
    retriever_output = [
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    ]
    context = retriever_output if retriever_output else ["No relevant context found."]

    history_messages = []
    if history:
        for item in history:
            try:
                history_messages.append(Message(**item))
            except Exception:
                continue

    prompt = construct_prompt(query, context, history_messages)
    output, _provider_used = generate_with_fallback(
        provider,
        model,
        temperature,
        False,
        prompt,
    )
    return output


def query_backend(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    backend_url: str | None = None,
    retries: int = 1,
    **payload_kwargs: Any,
) -> str:
    """Call the backend and return the assistant answer."""
    history = history or []
    path, payload = build_rag_payload(query, history, **payload_kwargs)
    backend_url = backend_url or get_backend_url()
    local_backend = backend_url.rstrip("/") == get_backend_url().rstrip("/")

    if local_backend and path == "/generate/rag":
        try:
            return _local_rag_answer(
                query,
                history,
                provider=payload.get("provider"),
                model=payload.get("model"),
                temperature=payload.get("temperature"),
                source=payload.get("source"),
                k=payload.get("k"),
            )
        except Exception as exc:
            raise RuntimeError(f"Local backend generation failed: {exc}") from exc

    url = f"{backend_url}{path}"
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = httpx.post(url, json=payload, timeout=CLIENT_REQUEST_TIMEOUT)
        except httpx.ConnectError as exc:
            last_error = RuntimeError(f"Could not connect to backend at {backend_url}: {exc}")
            if attempt < retries:
                continue
            raise last_error from exc
        except httpx.TimeoutException as exc:
            last_error = RuntimeError(f"Request timed out calling {path}")
            if attempt < retries:
                continue
            raise last_error from exc

        if response.status_code < 400:
            data = response.json()
            return data.get("answer", "")

        last_error = RuntimeError(
            f"{path} returned {response.status_code}: {response.text[:500]}"
        )
        if attempt < retries:
            continue
        raise last_error

    raise RuntimeError("Backend request failed")


def check_backend(backend_url: str | None = None) -> bool:
    url = f"{(backend_url or get_backend_url())}/docs"
    try:
        response = httpx.get(url, timeout=5.0)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


def warmup_ollama(model: str | None = None) -> None:
    """Load the Ollama model before the first user question."""
    if (LLM_PROVIDER or "").lower() != "ollama":
        return

    model_name = _resolve_model("ollama", model or DEFAULT_LLM_MODEL) or "llama3"
    print(f"Warming up Ollama model '{model_name}' (can take 10-20s)...", flush=True)
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            print(
                f"Ollama warmup failed ({response.status_code}): {response.text[:200]}",
                flush=True,
            )
            print(
                f"Pull the model with: ollama pull {model_name}",
                flush=True,
            )
            return
        print("Ollama ready.", flush=True)
    except httpx.HTTPError as exc:
        print(f"Ollama warmup failed: {exc}", flush=True)
        print("Start Ollama with: ollama serve", flush=True)
