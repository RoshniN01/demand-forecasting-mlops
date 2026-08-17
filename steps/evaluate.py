from zenml import step

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


@step
def evaluate(
    result: pd.DataFrame,
) -> None:

    actual = result["Actual"]
    predicted = result["Predicted"]

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # MAPE
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

    print("\n============================")
    print("FORECAST RESULTS")
    print("============================")

    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    print("\nActual vs Predicted:")
    print(result.head(10))