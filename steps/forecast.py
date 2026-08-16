from zenml import step
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor


@step
def forecast(
    ts: pd.DataFrame,
    model: LGBMRegressor,
) -> pd.DataFrame:

    print("Generating forecasts...")

    features = [
        "day_number",
        "lag_1",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "day_of_week",
        "month",
    ]

    # Last 28 days are the test/forecast period
    test_data = ts.iloc[-28:].copy()

    X_test = test_data[features]

    predictions = model.predict(X_test)

    # Demand cannot be negative
    predictions = np.maximum(predictions, 0)

    result = pd.DataFrame({
        "Actual": test_data["sales"].values,
        "Predicted": predictions,
    })

    print("Forecasting completed.")
    print("Forecast rows:", len(result))

    return result