import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND_URL = os.getenv("BACKEND", "http://localhost:8000")

DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
DEFAULT_VECTOR_SOURCE_LABEL = os.getenv("DEFAULT_VECTOR_SOURCE_LABEL", "system-source")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))
DEFAULT_RAG_K = int(os.getenv("DEFAULT_RAG_K", "5"))
DEFAULT_RAG_SOURCE_FILTER = os.getenv("DEFAULT_RAG_SOURCE_FILTER", "")
DEFAULT_USE_RAG = os.getenv("DEFAULT_USE_RAG", "true").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
CLIENT_REQUEST_TIMEOUT = int(os.getenv("CLIENT_REQUEST_TIMEOUT", "600"))
