def preprocess_input(transaction: dict) -> dict:
    """
    Preprocess single transaction for H2O Mojo
    """

    data = transaction.copy()

    # Define categorical columns that should be converted to strings
    # Note: C1-C14 are count features and should remain numeric
    categorical_cols = [
        'ProductCD', 'card4', 'card6',
        'P_emaildomain', 'R_emaildomain',
        'DeviceType', 'DeviceInfo',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
    ]
    
    # Convert categorical columns to strings
    for col in categorical_cols:
        if col in data and data[col] is not None:
            data[col] = str(data[col])
    
    # Fill missing values
    for key, value in data.items():
        if value is None:
            # Use "Unknown" for categorical columns, 0 for numeric
            if key in categorical_cols:
                data[key] = "Unknown"
            else:
                data[key] = 0

    # Feature: large transaction
    if "TransactionAmt" in data:
        data["is_large_amount"] = int(data["TransactionAmt"] > 1000)
    else:
        data["is_large_amount"] = 0

    # Feature: transaction count
    # Cannot compute groupby in API → safe default
    data["transaction_count"] = 1

    return data

