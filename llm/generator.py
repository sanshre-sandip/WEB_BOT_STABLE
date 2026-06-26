import logging
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    GOOGLE_API_KEY, OLLAMA_BASE_URL, OLLAMA_REQUEST_TIMEOUT
)

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from RAG.vectorstore import get_vectorstore
except ImportError:
    get_vectorstore = None

router = APIRouter()
logger = logging.getLogger(__name__)

FALLBACK_PROVIDER = "ollama"


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    query: str
    context: List[str]
    history: Optional[List[Message]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class RagRequest(BaseModel):
    query: str
    source: Optional[str] = None
    k: int = 5
    history: Optional[List[Message]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class GenerateResponse(BaseModel):
    status: str
    answer: str
    model_used: str
    provider_used: str


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or LLM_PROVIDER or FALLBACK_PROVIDER).lower()


def resolve_model(provider: Optional[str], model: Optional[str]) -> Optional[str]:
    """Ignore empty values and mistaken provider names used as model names."""
    if not model or not model.strip():
        return None
    normalized_model = model.strip()
    if normalized_model.lower() == normalize_provider(provider):
        return None
    return normalized_model


def get_provider_order(provider: Optional[str]) -> list[str]:
    normalized = normalize_provider(provider)

    if normalized == "auto":
        return ["openai", "anthropic", "google", FALLBACK_PROVIDER]

    if normalized == FALLBACK_PROVIDER:
        return [FALLBACK_PROVIDER]

    return [normalized, FALLBACK_PROVIDER]


def get_llm(provider: str, model: Optional[str], temperature: float = 0.7, streaming: bool = False):
    provider = provider.lower()

    if provider == "openai":
        if not ChatOpenAI:
            raise HTTPException(status_code=500, detail="langchain-openai not installed")
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment")
        return ChatOpenAI(model=model or "gpt-4o", api_key=OPENAI_API_KEY, temperature=temperature, streaming=streaming)

    if provider == "anthropic":
        if not ChatAnthropic:
            raise HTTPException(status_code=500, detail="langchain-anthropic not installed")
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found in environment")
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY, temperature=temperature, streaming=streaming)

    if provider == "google":
        if not ChatGoogleGenerativeAI:
            raise HTTPException(status_code=500, detail="langchain-google-genai not installed")
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not found in environment")
        return ChatGoogleGenerativeAI(model=model or "gemini-1.5-pro", google_api_key=GOOGLE_API_KEY, temperature=temperature, streaming=streaming)

    if provider == "ollama":
        if not ChatOllama:
            raise HTTPException(status_code=500, detail="langchain-ollama or langchain-community not installed")
        return ChatOllama(model=model or "llama3", base_url=OLLAMA_BASE_URL, temperature=temperature, streaming=streaming, request_timeout=OLLAMA_REQUEST_TIMEOUT)

    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")


def extract_content(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


def generate_with_fallback(
    provider: Optional[str],
    model: Optional[str],
    temperature: float,
    streaming: bool,
    prompt: str,
) -> tuple[Any, str]:
    errors = []

    for candidate_provider in get_provider_order(provider):
        try:
            llm = get_llm(candidate_provider, model, temperature, streaming)

            if streaming:
                async def stream_generator():
                    async for chunk in llm.astream(prompt):
                        yield extract_content(chunk)

                return stream_generator(), candidate_provider

            response = llm.invoke(prompt)
            return extract_content(response), candidate_provider

        except Exception as exc:
            errors.append(f"{candidate_provider}: {exc}")

    raise HTTPException(status_code=500, detail="All LLM providers failed: " + " | ".join(errors))


def construct_prompt(query: str, context: List[str], history: Optional[List[Message]] = None):
    system_msg = """You are Aasiii, a friendly assistant for Aasiii Tech.
Answer only using the retrieved context provided to you.

## Strict Rules

1. **Never mention the retriever, context, or backend system** — the user should never know how you work internally. Never say phrases like "the retriever output", "based on the context", "retrieved information", or anything similar.
2. **If the question is outside your knowledge**, respond naturally like:
   "That's a great question! I don't have enough details on that right now — I'd recommend reaching out to our team directly and they'll sort you out! 😊"
3. **Never guess or fabricate** — if it's not in your knowledge base, use the fallback above.
4. **Stay on topic** — you help with questions related to Aasiii Tech's services, products, and company info only.

## Tone & Style

- Friendly, casual, and warm — like a helpful teammate.
- Short and clear responses (2–4 sentences for simple questions).
- Use bullet points for multi-step or detailed answers.
- Never sound robotic or corporate.

## What You Must NEVER Say

❌ "The retriever output says..."
❌ "Based on the provided context..."
❌ "The retriever only discusses..."
❌ "I don't have access to information about that topic."

## What You Should Say Instead

✅ "I don't have that info handy — our team can help though!"
✅ "Great question! For that one, I'd suggest contacting us directly."
✅ [Answer naturally if the info is available]"""

    context_text = "\n\n---\n\n".join(context)

    prompt = f"{system_msg}\n\nRetriever Output:\n{context_text}\n\n"

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            prompt += f"{msg.role.capitalize()}: {msg.content}\n"
        prompt += "\n"

    prompt += f"User Query: {query}\n"
    prompt += "Assistant Answer:"

    return prompt


@router.post("", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    provider = request.provider or LLM_PROVIDER
    model = resolve_model(provider, request.model)
    prompt = construct_prompt(request.query, request.context, request.history)

    try:
        output, provider_used = generate_with_fallback(
            provider,
            model,
            request.temperature,
            request.stream,
            prompt,
        )

        logger.info(f"Generating answer using {provider_used}...")

        if request.stream:
            return StreamingResponse(output, media_type="text/plain")

        return GenerateResponse(
            status="success",
            answer=output,
            model_used=model or "default",
            provider_used=provider_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=GenerateResponse)
async def rag_answer(request: RagRequest):
    if get_vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized or RAG modules not found")

    provider = request.provider or LLM_PROVIDER
    model = resolve_model(provider, request.model)

    try:
        logger.info(f"Retrieving context for query: {request.query}")
        store = get_vectorstore()
        filter_dict = {"source": request.source} if request.source else None
        docs = store.similarity_search(request.query, k=request.k, filter=filter_dict)
        retriever_output = [
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        ]
        context = retriever_output if retriever_output else ["No relevant context found."]

        prompt = construct_prompt(request.query, context, request.history)

        output, provider_used = generate_with_fallback(
            provider,
            model,
            request.temperature,
            request.stream,
            prompt,
        )

        logger.info(f"Generating RAG answer using {provider_used}...")

        if request.stream:
            return StreamingResponse(output, media_type="text/plain")

        return GenerateResponse(
            status="success",
            answer=output,
            model_used=model or "default",
            provider_used=provider_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG Generation failed")
        raise HTTPException(status_code=500, detail=str(e))
