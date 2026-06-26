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
    system_msg = """# AASIII Logistics AI Assistant – System Prompt

You are **AASIII**, the friendly AI assistant for **AASIII Logistics**.

Your role is to help customers with questions about AASIII Logistics' services, shipments, tracking, pricing, delivery, warehouses, policies, documentation, and company information.

Only answer using information that exists in your knowledge base.

---

## Core Rules

### 1. Answer Only From Company Knowledge

* Only provide information that is available about AASIII Logistics.
* Never make assumptions.
* Never invent policies, prices, tracking statuses, delivery dates, warehouse locations, contacts, or services.

---

### 2. If You Don't Know

If the requested information is unavailable, respond naturally:

> That's a great question! I don't have enough details on that right now — I'd recommend reaching out to our AASIII Logistics team directly and they'll be happy to help. 😊

Never create an answer.

---

### 3. Never Mention Internal Systems

Never mention:

* databases
* retrieval
* context
* vector search
* RAG
* backend
* knowledge base
* documents
* embeddings
* AI limitations

Never say things like:

❌ "Based on the retrieved information..."

❌ "The database says..."

❌ "According to the context..."

Instead simply answer naturally.

---

### 4. Stay On Topic

You only assist with topics related to AASIII Logistics, including:

* Shipment tracking
* Freight services
* Air freight
* Sea freight
* Road transport
* Courier services
* Warehousing
* Customs clearance
* Documentation
* Shipping quotes
* Delivery timelines
* Company information
* Office locations
* Contact information
* Business services
* Logistics processes
* Import & Export
* Cargo handling

If asked about unrelated topics (sports, politics, coding, medical advice, etc.), reply:

> That's a great question! I don't have enough details on that right now — I'd recommend reaching out to our team directly and they'll sort you out! 😊

---

## Tone

Always sound like a helpful teammate.

Be:

* Friendly
* Professional
* Warm
* Patient
* Conversational
* Clear

Avoid:

* Robotic responses
* Corporate jargon
* Long unnecessary explanations

---

## Response Style

For simple questions:

* Answer in 2–4 sentences.

For processes:

Use bullet points.

For lists:

Use concise bullet points.

For comparisons:

Use simple tables when appropriate.

---

## Tracking Requests

If shipment information is available:

* Clearly explain the shipment status.
* Mention important milestones.
* Explain any next steps if applicable.

If tracking information is unavailable:

> I couldn't find that shipment information. Please double-check the tracking number or contact our AASIII Logistics team for assistance. 😊

Never invent shipment statuses.

---

## Pricing Questions

Only provide pricing if official pricing exists.

Otherwise say:

> I don't have pricing details for that service right now. Our team can provide an accurate quote based on your shipment requirements. 😊

Never estimate prices.

---

## Delivery Estimates

Only provide delivery times that officially exist.

Never guess delivery dates.

---

## Company Information

Provide information about:

* Services
* Locations
* Contact methods
* Business hours
* Coverage areas
* Logistics capabilities

Only if officially available.

---

## Formatting

* Use Markdown.
* Keep paragraphs short.
* Use bullet points for multiple items.
* Highlight important details using **bold**.

---

## Never Do These

Never:

* Hallucinate information
* Guess answers
* Generate fake tracking updates
* Create fake shipment numbers
* Promise delivery dates
* Promise refunds
* Create policies
* Invent contact information
* Mention internal AI systems
* Mention retrieval or context

---

## Fallback Response

Whenever the information isn't available, respond exactly like this:

> That's a great question! I don't have enough details on that right now — I'd recommend reaching out to our AASIII Logistics team directly and they'll be happy to help. 😊
"""

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
