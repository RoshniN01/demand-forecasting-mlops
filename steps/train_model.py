from zenml import step

import os
import pandas as pd
import numpy as np
import optuna
import mlflow
import joblib

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


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
def train_model(data: pd.DataFrame) -> str:

    print("Preparing training data...")

    X = data[FEATURES]
    y = data["sales"]

    # Last 28 days are used for testing
    test_size = 28

    X_train = X.iloc[:-test_size]
    y_train = y.iloc[:-test_size]

    X_test = X.iloc[-test_size:]
    y_test = y.iloc[-test_size:]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # MLflow experiment
    mlflow.set_experiment("demand_forecasting")

    # Optuna
    print("\nStarting Optuna tuning...")

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 400
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 20, 60
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 10
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 50
            ),
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # Record this Optuna trial in MLflow
        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}",
            nested=True
        ):

            mlflow.log_params({
                "trial_number": trial.number,
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "max_depth": params["max_depth"],
                "min_child_samples": params["min_child_samples"],
            })

            mlflow.log_metric(
                "trial_rmse",
                rmse
            )

        print(
            f"Trial {trial.number}: "
            f"RMSE = {rmse:.4f}"
        )

        return rmse

    # Run Optuna inside MLflow run
    with mlflow.start_run(
        run_name="optuna_lightgbm"
    ):

        study = optuna.create_study(
            direction="minimize",
            study_name="lightgbm_demand_forecasting"
        )

        study.optimize(
            objective,
            n_trials=10
        )

        print("\nBest parameters:")
        print(study.best_params)

        print("\nBest Optuna RMSE:")
        print(study.best_value)

        # Log best parameters
        mlflow.log_params(
            study.best_params
        )

        # Log Optuna results
        mlflow.log_metric(
            "best_optuna_rmse",
            study.best_value
        )

        mlflow.log_metric(
            "number_of_trials",
            len(study.trials)
        )

        # Train final model
        model = LGBMRegressor(
            **study.best_params
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0
        )

        # Calculate RMSE
        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # Calculate MAPE
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

        # Log final metrics
        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mape",
            mape
        )

        print("\nFinal Model Results")
        print("-------------------")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    # Save model
    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/forecast_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved to:",
        model_path
    )

    return model_path