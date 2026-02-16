import sys
import os

# Add project root to Python path (needed when running from different contexts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import h2o
import json
import pandas as pd
import warnings
from src.data_pipeline.preprocess import preprocess_input

# Suppress H2O warnings about missing columns
warnings.filterwarnings('ignore', category=UserWarning, module='h2o')
warnings.filterwarnings('ignore', category=Warning, module='h2o')

# Initialize H2O once - start local server
h2o.init(max_mem_size="2G", start_h2o=True, strict_version_check=False)

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

    # SHAP Contributions
    contrib = model.predict_contributions(hf)
    contrib_df = contrib.as_data_frame()

    # Remove bias term
    contrib_df = contrib_df.drop(columns=["BiasTerm"], errors="ignore")

    # Sort by absolute contribution
    contrib_series = contrib_df.iloc[0].abs().sort_values(ascending=False)

    # Get top 3 features
    top_features = contrib_series.head(3)

    return {
        "fraud_probability": float(result["p1"][0]),
        "prediction": int(result["predict"][0]),
        "top_features": top_features.to_dict()
    }
