import h2o
import json
import os
import pandas as pd
from data_pipeline.preprocess import preprocess_input

# Initialize H2O once
h2o.init(max_mem_size="2G")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../models_artifacts/XGBoost_1_AutoML_1_20260209_165338.zip"
)

model = h2o.import_mojo(MODEL_PATH)


def predict_transaction(transaction: dict):
    """
    Takes a single transaction dict and returns fraud probability
    """
    processed_transaction = preprocess_input(transaction)

    # Convert to pandas DataFrame first to control data types
    df = pd.DataFrame([processed_transaction])
    
    # Ensure numeric columns are properly typed (not categorical)
    numeric_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                    'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                    'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 
                    'D11', 'D12', 'D13', 'D14', 'D15',
                    'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
                    'is_large_amount', 'transaction_count']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert to H2OFrame
    hf = h2o.H2OFrame(df)
    preds = model.predict(hf)
    result = preds.as_data_frame()

    return {
        "fraud_probability": float(result["p1"][0]),
        "prediction": int(result["predict"][0])
    }
