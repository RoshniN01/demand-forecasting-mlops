from zenml import step
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


@step
def evaluate(
    result: pd.DataFrame,
) -> None:

    print("\nEvaluating forecast...")

    actual = result["Actual"]
    predicted = result["Predicted"]

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # MAPE calculated only where actual demand is non-zero
    mask = actual != 0

    if mask.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100
    else:
        mape = np.nan

    print("\n==============================")
    print("ZENML FORECAST RESULTS")
    print("==============================")

    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

    print("\nActual vs Predicted:")
    print(result.head(10))