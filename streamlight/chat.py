import sys
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
    CLIENT_REQUEST_TIMEOUT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_RAG_K,
    DEFAULT_RAG_SOURCE_FILTER,
    DEFAULT_USE_RAG,
    LLM_PROVIDER,
)

load_dotenv()

st.set_page_config(page_title="Web Bot Chat", layout="wide")

DEFAULT_TIMEOUT = CLIENT_REQUEST_TIMEOUT


def get_backend_url() -> str:
    return BACKEND_URL.rstrip("/")


def call_backend(path: str, payload: dict[str, Any], backend_url: str) -> dict[str, Any]:
    url = f"{backend_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to backend at {backend_url}") from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Request timed out while calling {path}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned {response.status_code}: {response.text[:500]}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Response from {path} was not valid JSON: {response.text[:500]}") from exc


def build_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("backend_url", get_backend_url())


def render_sidebar() -> tuple[str, str, float, bool, str, int]:
    st.sidebar.title("System Chat Settings")
    st.sidebar.caption(f"Backend: {get_backend_url()}")
    st.sidebar.caption(f"Request timeout: {CLIENT_REQUEST_TIMEOUT}s")
    st.sidebar.caption(f"Provider: {LLM_PROVIDER or 'backend default'}")
    st.sidebar.caption(f"Model: {DEFAULT_LLM_MODEL or 'backend default'}")
    st.sidebar.caption(f"Temperature: {DEFAULT_LLM_TEMPERATURE}")
    st.sidebar.caption(f"RAG enabled: {DEFAULT_USE_RAG}")
    st.sidebar.caption(f"RAG top_k: {DEFAULT_RAG_K}")
    st.sidebar.caption(f"RAG source filter: {DEFAULT_RAG_SOURCE_FILTER or 'none'}")

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    provider = LLM_PROVIDER or "backend default"
    return get_backend_url(), provider, DEFAULT_LLM_TEMPERATURE, DEFAULT_USE_RAG, DEFAULT_RAG_SOURCE_FILTER, DEFAULT_RAG_K


def send_chat_message(
    backend_url: str,
    query: str,
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    temperature: float,
    use_rag: bool,
    source: str,
    k: int,
) -> str:
    history = build_history(messages)

    if use_rag:
        payload: dict[str, Any] = {
            "query": query,
            "source": source or None,
            "k": k,
            "history": history,
            "temperature": temperature,
            "stream": False,
        }
        if provider != "backend default":
            payload["provider"] = provider
        if model:
            payload["model"] = model
        data = call_backend("/generate/rag", payload, backend_url)
        return data.get("answer", "")

    payload = {
        "query": query,
        "context": ["No retrieval context provided."],
        "history": history,
        "temperature": temperature,
        "stream": False,
    }
    if provider != "backend default":
        payload["provider"] = provider
    if model:
        payload["model"] = model

    data = call_backend("/generate", payload, backend_url)
    return data.get("answer", "")


def render_messages() -> None:
    if not st.session_state.messages:
        st.info("Ask a question below to start chatting.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def main() -> None:
    initialize_session_state()
    backend_url, provider, temperature, use_rag, source, k = render_sidebar()

    st.title("Web Bot Chat")
    st.caption("Chat with the FastAPI backend using system-defined chat settings.")

    render_messages()

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking...")
        try:
            answer = send_chat_message(
                backend_url,
                prompt,
                st.session_state.messages[:-1],
                provider,
                DEFAULT_LLM_MODEL,
                temperature,
                use_rag,
                source,
                k,
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            placeholder.write(answer)
        except Exception as exc:
            error_message = str(exc)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            placeholder.error(error_message)

    st.rerun()


if __name__ == "__main__":
    main()
