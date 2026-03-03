"""
Integration Tests for Fraud Detection System.

Tests end-to-end flow including API and model interaction.
These tests require H2O model to be available (slower, marked as integration).
"""

import pytest
import httpx
import asyncio
from pathlib import Path

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def api_base_url():
    """API base URL for testing."""
    return "http://localhost:8000"


@pytest.fixture
async def http_client(api_base_url):
    """Create async HTTP client for API calls."""
    async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    async def test_api_health_check(self, http_client):
        """
        Test API health check endpoint.
        
        Requires: API server running at localhost:8000
        """
        response = await http_client.get("/health")
        assert response.status_code in [200, 503]  # May be unavailable if model not loaded
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "h2o_connected" in data
    
    async def test_prediction_end_to_end(self, http_client):
        """
        Test complete prediction flow from request to response.
        
        Requires: API server running with loaded model
        """
        transaction = {
            "TransactionAmt": 200075.50,
            "card1": 19945,
            "addr1": 19999,
            "DeviceType": "mobile",
            "C1": 3,
            "D1": 25.0
        }
        
        response = await http_client.post("/predict", json=transaction)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify response structure
        assert "fraud_probability" in data
        assert "prediction" in data
        assert "risk_level" in data
        assert "top_features" in data
        assert "request_id" in data
        assert "timestamp" in data
        assert "model_version" in data
        
        # Verify data types and ranges
        assert isinstance(data["fraud_probability"], float)
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["prediction"] in [0, 1]
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert isinstance(data["top_features"], list)
        assert len(data["top_features"]) <= 3
        
        # Verify top features structure
        for feature in data["top_features"]:
            assert "feature" in feature
            assert "contribution" in feature
            assert isinstance(feature["contribution"], (int, float))
    
    async def test_multiple_predictions_consistency(self, http_client):
        """
        Test that multiple predictions for the same transaction are consistent.
        
        Requires: API server running with loaded model
        """
        transaction = {
            "TransactionAmt": 150.00,
            "card1": 12345
        }
        
        # Make multiple predictions
        responses = []
        for _ in range(3):
            response = await http_client.post("/predict", json=transaction)
            responses.append(response.json())
        
        # All should succeed
        assert all(r["prediction"] == responses[0]["prediction"] for r in responses)
        assert all(r["fraud_probability"] == responses[0]["fraud_probability"] for r in responses)
    
    async def test_prediction_performance(self, http_client):
        """
        Test prediction response time (should be < 5 seconds).
        
        Requires: API server running with loaded model
        """
        import time
        
        transaction = {
            "TransactionAmt": 100.00,
            "card1": 1000
        }
        
        start_time = time.time()
        response = await http_client.post("/predict", json=transaction)
        duration = time.time() - start_time
        
        assert response.status_code == 200
        assert duration < 5.0, f"Prediction took {duration:.2f}s, expected < 5.0s"


class TestModelIntegration:
    """Integration tests for model predictor (requires H2O)."""
    
    @pytest.mark.slow
    def test_predictor_initialization(self):
        """
        Test that predictor can initialize H2O and load model.
        
        This is slow as it starts H2O server.
        """
        from src.models.predictor import get_predictor
        
        predictor = get_predictor()
        health = predictor.health_check()
        
        # After warmup, model should be loaded
        predictor.warmup()
        health_after = predictor.health_check()
        
        assert health_after["model_loaded"] is True
        assert health_after["h2o_initialized"] is True
    
    @pytest.mark.slow
    def test_predictor_prediction(self):
        """
        Test predictor can make predictions.
        
        This is slow as it may start H2O server.
        """
        from src.models.predictor import predict_transaction
        
        transaction = {
            "TransactionAmt": 200075.50,
            "card1": 19945
        }
        
        result = predict_transaction(transaction)
        
        assert "fraud_probability" in result
        assert "prediction" in result
        assert "top_features" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["prediction"] in [0, 1]


class TestDashboardAPIIntegration:
    """Integration tests for dashboard → API communication."""
    
    async def test_dashboard_can_call_api(self, http_client):
        """
        Test that dashboard's API client can successfully call prediction endpoint.
        
        Simulates dashboard behavior.
        """
        # This is what the dashboard does
        transaction = {
            "TransactionAmt": 150.00,
            "card1": 12345
        }
        
        try:
            response = await http_client.post("/predict", json=transaction)
            response.raise_for_status()
            data = response.json()
            
            # Verify dashboard can process the response
            assert "fraud_probability" in data
            assert "top_features" in data
            
            # Dashboard expects top_features as list
            assert isinstance(data["top_features"], list)
            for feature in data["top_features"]:
                assert "feature" in feature
                assert "contribution" in feature
                
        except httpx.RequestError as e:
            pytest.skip(f"API not available: {e}")


class TestErrorHandling:
    """Integration tests for error handling."""
    
    async def test_api_handles_invalid_input(self, http_client):
        """Test that API properly handles invalid input."""
        invalid_transaction = {
            "TransactionAmt": -100.00,  # Negative amount
            "card1": 12345
        }
        
        response = await http_client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data
    
    async def test_api_handles_missing_fields(self, http_client):
        """Test that API properly handles missing required fields."""
        incomplete_transaction = {
            "card1": 12345
            # Missing TransactionAmt
        }
        
        response = await http_client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422


# Utility function to check if API is ready for integration tests
async def is_api_running(base_url: str = "http://localhost:8000") -> bool:
    """Check if API server is reachable and model is ready."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            response = await client.get("/health")
            if response.status_code != 200:
                return False

            data = response.json()
            return bool(data.get("model_loaded")) and bool(data.get("h2o_connected"))
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_api_availability():
    """
    Check if API is available before running integration tests.
    
    Skip all integration tests if API is not running.
    """
    async def check():
        return await is_api_running()
    
    if not asyncio.run(check()):
        pytest.skip(
            "API server is not healthy/model-ready at localhost:8000. "
            "Start the API with 'python -m uvicorn src.api.main:app' "
            "or 'docker-compose up' before running integration tests.",
            allow_module_level=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
