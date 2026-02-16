import json
from src.models.predictor import predict_transaction

with open("src/api/transaction.json") as f:
    transaction = json.load(f)

result = predict_transaction(transaction)

print(result)
