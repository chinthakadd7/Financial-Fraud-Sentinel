import json
import sys
import os
import pytest

# Make src importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.predictor import predict_transaction


def test_predict_transaction_with_sample_data():
    """Test prediction with sample transaction from JSON file"""
    with open("src/api/transaction.json", "r") as f:
        transaction = json.load(f)

    result = predict_transaction(transaction)
    
    # Assert the result has expected keys
    assert "fraud_probability" in result, "Result should contain fraud_probability"
    assert "prediction" in result, "Result should contain prediction"
    
    # Assert values are in valid ranges
    assert 0 <= result["fraud_probability"] <= 1, "Fraud probability should be between 0 and 1"
    assert result["prediction"] in [0, 1], "Prediction should be 0 or 1"
    
    print(f"Prediction result: {result}")


def test_predict_transaction_with_mock_data():
    """Test prediction with mock transaction data"""
    mock_transaction = {
        "TransactionAmt": 500.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "credit"
    }
    
    result = predict_transaction(mock_transaction)
    
    assert "fraud_probability" in result
    assert "prediction" in result
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["prediction"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
