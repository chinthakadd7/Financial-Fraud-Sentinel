import json
import sys
import os

# Make src importable
sys.path.append(os.path.abspath("src"))

from models.predictor import predict_transaction


def main():
    with open("src/api/transaction.json", "r") as f:
        transaction = json.load(f)

    result = predict_transaction(transaction)
    print(result)


if __name__ == "__main__":
    main()
