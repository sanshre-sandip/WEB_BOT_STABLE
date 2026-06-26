import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from RAG.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_db_info_endpoint(client):
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": ["1", "2"],
        "metadatas": [{"source": "A"}, {"source": "B"}]
    }

    with patch("RAG.db.get_vectorstore", return_value=mock_store):
        response = client.get("/db/info")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 2
        assert data["unique_sources"] == 2

def test_list_chunks_endpoint(client):
    mock_store = MagicMock()
    mock_store.get.return_value = {
        "ids": ["1"],
        "documents": ["doc1"],
        "metadatas": [{"source": "A"}]
    }
    # Mock total matching check
    mock_store.get.side_effect = [
        {"ids": ["1"], "documents": ["doc1"], "metadatas": [{"source": "A"}]},
        {"ids": ["1"]}
    ]

    with patch("RAG.db.get_vectorstore", return_value=mock_store):
        response = client.get("/db/list")
        assert response.status_code == 200
        data = response.json()
        assert data["returned"] == 1
        assert data["chunks"][0]["source"] == "A"
