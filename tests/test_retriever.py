import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from RAG.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_query_endpoint(client):
    mock_store = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "doc1"
    mock_doc.metadata = {"id": "1", "source": "A"}
    mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.9)]

    with patch("RAG.retriever.get_vectorstore", return_value=mock_store):
        response = client.post("/retrieve/query", json={"query": "test query", "k": 1})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["results"][0]["text"] == "doc1"
