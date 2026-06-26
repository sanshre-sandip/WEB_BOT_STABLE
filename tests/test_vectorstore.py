import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from RAG.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_search_endpoint(client):
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["1"]}
    mock_doc = MagicMock()
    mock_doc.page_content = "test content"
    mock_doc.metadata = {"source": "test_source"}
    mock_store.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]

    with patch("RAG.vectorstore.get_vectorstore", return_value=mock_store):
        response = client.post("/vectorstore/search", json={"query": "test query", "top_k": 1})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
