import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# 1. Load M5 sales data
# --------------------------------------------------

print("Loading sales data...")

sales = pd.read_csv("data/sales_train_evaluation.csv")

print("Sales data shape:", sales.shape)


# --------------------------------------------------
# 2. Select one product/store series
# --------------------------------------------------

# We use one representative Walmart product/store
# to keep today's training fast and manageable.

row = sales.iloc[0]

id_value = row["id"]
item_id = row["item_id"]
store_id = row["store_id"]

print("\nSelected series:")
print("ID:", id_value)
print("Item:", item_id)
print("Store:", store_id)


# --------------------------------------------------
# 3. Convert daily columns into a time series
# --------------------------------------------------

d_cols = [c for c in sales.columns if c.startswith("d_")]

ts = pd.DataFrame({
    "day": d_cols,
    "sales": pd.to_numeric(row[d_cols].values, errors="coerce")
})

ts["day_number"] = np.arange(len(ts))


# --------------------------------------------------
# 4. Create time-series features
# --------------------------------------------------

ts["lag_1"] = ts["sales"].shift(1)
ts["lag_7"] = ts["sales"].shift(7)
ts["lag_28"] = ts["sales"].shift(28)

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

# Simple calendar features
ts["day_of_week"] = ts["day_number"] % 7
ts["month"] = (ts["day_number"] // 30) % 12 + 1


# Remove rows created by lagging
ts = ts.dropna().reset_index(drop=True)


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

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

# Last 28 days = test set
split = len(ts) - 28

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 6. Optuna hyperparameter tuning
# --------------------------------------------------


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 60
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 50
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "random_state": 42,
        "verbosity": -1
    }

    model = LGBMRegressor(**params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    return rmse


mlflow.set_experiment("demand_forecasting")
mlflow.start_run(run_name="optuna_lightgbm")
print("\nStarting Optuna hyperparameter tuning...")

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=10
)

print("\nBest Optuna parameters:")
print(study.best_params)

print("Best RMSE:", study.best_value)

mlflow.log_params(study.best_params)
mlflow.log_metric("best_optuna_rmse", study.best_value)


# --------------------------------------------------
# 7. Train final tuned model
# --------------------------------------------------

print("\nTraining final tuned LightGBM model...")

best_model = LGBMRegressor(
    **study.best_params
)

best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)

predictions = np.maximum(predictions, 0)


# --------------------------------------------------
# 8. Final evaluation
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

mask = y_test != 0

if mask.sum() > 0:
    mape = np.mean(
        np.abs(
            (y_test[mask] - predictions[mask])
            / y_test[mask]
        )
    ) * 100
else:
    mape = np.nan


print("\n==============================")
print("FINAL TUNED MODEL RESULTS")
print("==============================")

print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

print("\nActual vs Predicted:")

print(
    pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }).head(10)
)

mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mape", mape)

mlflow.log_param("item_id", item_id)
mlflow.log_param("store_id", store_id)

mlflow.end_run()