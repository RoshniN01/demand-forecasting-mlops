from zenml import step
import pandas as pd
import numpy as np


@step
def feature_engineering() -> pd.DataFrame:

    print("Loading sales data...")

    sales = pd.read_csv("data/sales_train_evaluation.csv")

    print("Sales data shape:", sales.shape)

    # Select one representative product/store series
    row = sales.iloc[0]

    print("\nSelected series:")
    print("ID:", row["id"])
    print("Item:", row["item_id"])
    print("Store:", row["store_id"])

    # Get daily sales columns
    d_cols = [c for c in sales.columns if c.startswith("d_")]

    # Convert daily sales into a time series
    ts = pd.DataFrame({
        "day": d_cols,
        "sales": pd.to_numeric(
            row[d_cols].values,
            errors="coerce"
        )
    })

    ts["day_number"] = np.arange(len(ts))

    # Lag features
    ts["lag_1"] = ts["sales"].shift(1)
    ts["lag_7"] = ts["sales"].shift(7)
    ts["lag_28"] = ts["sales"].shift(28)

    # Rolling features
    ts["rolling_mean_7"] = (
        ts["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    ts["rolling_mean_28"] = (
        ts["sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    # Calendar features
    ts["day_of_week"] = ts["day_number"] % 7
    ts["month"] = (
        (ts["day_number"] // 30) % 12
    ) + 1

    # Remove rows created by lagging
    ts = ts.dropna().reset_index(drop=True)

    print("\nFeature engineering completed.")
    print("Feature data shape:", ts.shape)

    return ts