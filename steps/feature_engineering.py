from zenml import step

import pandas as pd
import numpy as np


ITEM_ID = "HOBBIES_1_001"
STORE_ID = "CA_1"


@step
def feature_engineering() -> pd.DataFrame:

    print("Loading M5 sales data...")

    sales = pd.read_csv(
        "data/sales_train_evaluation.csv"
    )

    # Select the product and store
    selected = sales[
        (sales["item_id"] == ITEM_ID)
        & (sales["store_id"] == STORE_ID)
    ]

    if selected.empty:
        raise ValueError(
            "The selected product and store were not found."
        )

    row = selected.iloc[0]

    print("Product:", row["item_id"])
    print("Store:", row["store_id"])

    # Get daily sales columns
    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    # Convert M5 wide data into time-series format
    data = pd.DataFrame({
        "day": day_columns,
        "sales": pd.to_numeric(
            row[day_columns].values,
            errors="coerce"
        )
    })

    data["day_number"] = np.arange(
        len(data)
    )

    # Previous-day sales
    data["lag_1"] = (
        data["sales"].shift(1)
    )

    # Sales one week ago
    data["lag_7"] = (
        data["sales"].shift(7)
    )

    # Sales four weeks ago
    data["lag_28"] = (
        data["sales"].shift(28)
    )

    # Average sales of previous 7 days
    data["rolling_mean_7"] = (
        data["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    # Average sales of previous 28 days
    data["rolling_mean_28"] = (
        data["sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    # Simple calendar features
    data["day_of_week"] = (
        data["day_number"] % 7
    )

    data["month"] = (
        (data["day_number"] // 30) % 12
    ) + 1

    # Remove rows with missing lag values
    data = data.dropna().reset_index(
        drop=True
    )

    print(
        "Feature engineering completed."
    )

    print(
        "Rows:",
        len(data)
    )

    return data