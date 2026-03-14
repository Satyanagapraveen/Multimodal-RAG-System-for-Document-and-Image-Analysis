import pytest
from fastapi.testclient import TestClient


# ── Setup ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Creates a TestClient for the FastAPI app."""
    from src.api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Health Endpoint Tests ────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """GET /health must return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must have all required fields."""
        response = client.get("/health")
        data = response.json()
        required_fields = ["status", "collections", "models_loaded", "message"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_health_status_is_healthy(self, client):
        """Health status must be 'healthy'."""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_collections_have_data(self, client):
        """Collections must have chunks indexed."""
        response = client.get("/health")
        collections = response.json()["collections"]
        assert collections["total"] > 0, \
            "No chunks in database — run ingestion first"

    def test_health_models_loaded(self, client):
        """Models must be loaded."""
        response = client.get("/health")
        assert response.json()["models_loaded"] is True


# ── Root Endpoint Tests ──────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        """GET / must return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_lists_endpoints(self, client):
        """Root must list available endpoints."""
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data


# ── Query Endpoint Tests ─────────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_query_returns_200(self, client):
        """POST /query with valid query must return 200."""
        response = client.post(
            "/query",
            json={"query": "What is attention mechanism?", "n_results": 3}
        )
        assert response.status_code == 200

    def test_query_response_structure(self, client):
        """Query response must have all required fields."""
        response = client.post(
            "/query",
            json={"query": "What is a transformer?", "n_results": 3}
        )
        data = response.json()
        required_fields = [
            "query", "answer", "sources",
            "response_time_seconds", "images_used",
            "context_chunks_used", "retrieval_breakdown"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_query_answer_not_empty(self, client):
        """Answer must not be empty."""
        response = client.post(
            "/query",
            json={"query": "Describe the BERT model", "n_results": 3}
        )
        data = response.json()
        assert len(data["answer"]) > 0

    def test_query_response_time_under_15s(self, client):
        """Response must complete under 15 seconds."""
        response = client.post(
            "/query",
            json={"query": "What is a GAN?", "n_results": 3}
        )
        data = response.json()
        assert data["response_time_seconds"] < 15.0, \
            f"Response too slow: {data['response_time_seconds']}s"

    def test_query_sources_is_list(self, client):
        """Sources must be a list."""
        response = client.post(
            "/query",
            json={"query": "What is word2vec?", "n_results": 3}
        )
        assert isinstance(response.json()["sources"], list)

    def test_query_empty_string_returns_400(self, client):
        """Empty query must return 400 Bad Request."""
        response = client.post(
            "/query",
            json={"query": "", "n_results": 3}
        )
        assert response.status_code == 400

    def test_query_whitespace_only_returns_400(self, client):
        """Whitespace-only query must return 400."""
        response = client.post(
            "/query",
            json={"query": "   ", "n_results": 3}
        )
        assert response.status_code == 400

    def test_query_too_long_returns_400(self, client):
        """Query over 1000 chars must return 400."""
        long_query = "a" * 1001
        response = client.post(
            "/query",
            json={"query": long_query, "n_results": 3}
        )
        assert response.status_code == 400

    def test_query_missing_body_returns_422(self, client):
        """Missing request body must return 422."""
        response = client.post("/query")
        assert response.status_code == 422

    def test_query_default_n_results(self, client):
        """Query without n_results must use default of 5."""
        response = client.post(
            "/query",
            json={"query": "transformer architecture"}
        )
        assert response.status_code == 200

    def test_query_multimodal_returns_images(self, client):
        """Visual query must return images_used > 0."""
        response = client.post(
            "/query",
            json={
                "query": "Describe the architecture diagram in the transformer paper",
                "n_results": 5
            }
        )
        data = response.json()
        assert data["images_used"] >= 0


# ── Ingest Endpoint Tests ────────────────────────────────────────────────────

class TestIngestEndpoint:

    def test_ingest_unsupported_type_returns_400(self, client):
        """Uploading a .txt file must return 400."""
        response = client.post(
            "/ingest",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        assert response.status_code == 400

    def test_ingest_missing_file_returns_422(self, client):
        """Missing file must return 422."""
        response = client.post("/ingest")
        assert response.status_code == 422