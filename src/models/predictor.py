import h2o
import json
import os

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
    hf = h2o.H2OFrame([transaction])
    preds = model.predict(hf)
    result = preds.as_data_frame()

    return {
        "fraud_probability": float(result["p1"][0]),
        "prediction": int(result["predict"][0])
    }
