from zenml import step
import pandas as pd
from lightgbm import LGBMRegressor


@step
def train_model(
    ts: pd.DataFrame,
) -> LGBMRegressor:

    print("Preparing training data...")

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

    X = ts[features]
    y = ts["sales"]

    # Last 28 days are kept for forecasting/evaluation
    split = len(ts) - 28

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    print("Training rows:", len(X_train))

    print("\nTraining LightGBM model...")

    model = LGBMRegressor(
        n_estimators=383,
        learning_rate=0.01611235188798151,
        num_leaves=45,
        max_depth=9,
        min_child_samples=21,
        subsample=0.7714093475207454,
        colsample_bytree=0.8122158282880609,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    print("LightGBM training completed.")

    return model