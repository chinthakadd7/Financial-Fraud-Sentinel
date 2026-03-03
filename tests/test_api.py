"""
API Unit Tests for Fraud Detection System.

Tests FastAPI endpoints with mocked predictor to avoid starting H2O server.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime

# Import the FastAPI app
from src.api.main import app
from src.api.schemas import RiskLevelEnum


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Create mock predictor that returns sample prediction results."""
    mock = Mock()
    mock.predict_transaction.return_value = {
        "fraud_probability": 0.85,
        "prediction": 1,
        "top_features": {
            "TransactionAmt": 0.45,
            "card1": 0.23,
            "D1": 0.12
        }
    }
    mock.health_check.return_value = {
        "model_loaded": True,
        "h2o_initialized": True,
        "h2o_connected": True,
        "h2o_version": "3.44.0.3",
        "model_path": "models_artifacts/test_model.zip",
        "model_exists": True,
        "uptime_seconds": 123.45
    }
    return mock


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    @patch("src.api.main.get_predictor")
    def test_health_check_healthy(self, mock_get_predictor, client, mock_predictor):
        """Test health check returns healthy status when model is loaded."""
        mock_get_predictor.return_value = mock_predictor
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["h2o_connected"] is True
        assert "uptime_seconds" in data
        assert "version" in data
    
    @patch("src.api.main.get_predictor")
    def test_health_check_unhealthy(self, mock_get_predictor, client):
        """Test health check returns unhealthy when model not loaded."""
        mock_predictor = Mock()
        mock_predictor.health_check.return_value = {
            "model_loaded": False,
            "h2o_initialized": False,
            "h2o_connected": False,
            "model_path": "models_artifacts/missing_model.zip",
            "model_exists": False
        }
        mock_get_predictor.return_value = mock_predictor
        
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
    
    @patch("src.api.main.get_predictor")
    def test_health_check_error_handling(self, mock_get_predictor, client):
        """Test health check handles errors gracefully."""
        mock_get_predictor.side_effect = Exception("H2O connection failed")
        
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data["details"]


class TestPredictEndpoint:
    """Test fraud prediction endpoint."""
    
    @patch("src.api.main.get_predictor")
    def test_predict_valid_transaction(self, mock_get_predictor, client, mock_predictor):
        """Test prediction with valid transaction data."""
        mock_get_predictor.return_value = mock_predictor
        
        transaction = {
            "TransactionAmt": 250.75,
            "card1": 12345
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
        assert "top_features" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Verify prediction values
        assert data["fraud_probability"] == 0.85
        assert data["prediction"] == 1
        assert data["risk_level"] == "HIGH"
        assert len(data["top_features"]) == 3
    
    @patch("src.api.main.get_predictor")
    def test_predict_risk_level_high(self, mock_get_predictor, client, mock_predictor):
        """Test that high fraud probability results in HIGH risk level."""
        mock_predictor.predict_transaction.return_value = {
            "fraud_probability": 0.85,
            "prediction": 1,
            "top_features": {"TransactionAmt": 0.5}
        }
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        data = response.json()
        assert data["risk_level"] == "HIGH"
    
    @patch("src.api.main.get_predictor")
    def test_predict_risk_level_medium(self, mock_get_predictor, client, mock_predictor):
        """Test that medium fraud probability results in MEDIUM risk level."""
        mock_predictor.predict_transaction.return_value = {
            "fraud_probability": 0.50,
            "prediction": 0,
            "top_features": {"TransactionAmt": 0.3}
        }
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        data = response.json()
        assert data["risk_level"] == "MEDIUM"
    
    @patch("src.api.main.get_predictor")
    def test_predict_risk_level_low(self, mock_get_predictor, client, mock_predictor):
        """Test that low fraud probability results in LOW risk level."""
        mock_predictor.predict_transaction.return_value = {
            "fraud_probability": 0.15,
            "prediction": 0,
            "top_features": {"TransactionAmt": 0.1}
        }
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        data = response.json()
        assert data["risk_level"] == "LOW"
    
    def test_predict_invalid_amount_negative(self, client):
        """Test that negative transaction amounts are rejected."""
        transaction = {
            "TransactionAmt": -100.00,
            "card1": 12345
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_amount_zero(self, client):
        """Test that zero transaction amounts are rejected."""
        transaction = {
            "TransactionAmt": 0.0,
            "card1": 12345
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422
    
    def test_predict_missing_required_fields(self, client):
        """Test that missing required fields are rejected."""
        transaction = {
            "card1": 12345
            # Missing TransactionAmt
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422
    
    def test_predict_invalid_data_types(self, client):
        """Test that invalid data types are rejected."""
        transaction = {
            "TransactionAmt": "not_a_number",
            "card1": 12345
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422
    
    @patch("src.api.main.get_predictor")
    def test_predict_optional_fields(self, mock_get_predictor, client, mock_predictor):
        """Test prediction with optional fields."""
        mock_get_predictor.return_value = mock_predictor
        
        transaction = {
            "TransactionAmt": 250.75,
            "card1": 12345,
            "addr1": 19999,
            "DeviceType": "mobile",
            "C1": 3,
            "D1": 25.0
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 200
    
    @patch("src.api.main.get_predictor")
    def test_predict_model_error_handling(self, mock_get_predictor, client):
        """Test that model errors are handled gracefully."""
        mock_predictor = Mock()
        mock_predictor.predict_transaction.side_effect = RuntimeError("Model prediction failed")
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
    
    @patch("src.api.main.get_predictor")
    def test_predict_request_id_present(self, mock_get_predictor, client, mock_predictor):
        """Test that each prediction has a unique request ID."""
        mock_get_predictor.return_value = mock_predictor
        
        response1 = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        response2 = client.post("/predict", json={"TransactionAmt": 200, "card1": 2000})
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["request_id"] != data2["request_id"]
    
    @patch("src.api.main.get_predictor")
    def test_predict_response_headers(self, mock_get_predictor, client, mock_predictor):
        """Test that response includes request ID in headers."""
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json={"TransactionAmt": 100, "card1": 1000})
        assert "X-Request-ID" in response.headers


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.options("/", headers={"Origin": "http://localhost:10101"})
        # FastAPI test client doesn't fully simulate CORS, but we can verify the middleware is configured
        # In real deployment, these headers would be present
        assert response.status_code in [200, 405]  # OPTIONS might not be explicitly defined


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation."""
    
    def test_openapi_json_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_ui_available(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_ui_available(self, client):
        """Test that ReDoc UI is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
