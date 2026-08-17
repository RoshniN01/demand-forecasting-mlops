from zenml import step

import pandas as pd
import numpy as np
import joblib


FEATURES = [
    "day_number",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
]


@step
def forecast(
    data: pd.DataFrame,
    model_path: str,
) -> pd.DataFrame:

    print("Generating forecast...")

    model = joblib.load(model_path)

    # Last 28 days are our test period
    test_data = data.iloc[-28:].copy()

    X_test = test_data[FEATURES]

    predictions = model.predict(X_test)

    # Demand cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    result = pd.DataFrame({
        "day": test_data["day"].values,
        "Actual": test_data["sales"].values,
        "Predicted": predictions,
    })

    print("Forecast completed.")

    return result