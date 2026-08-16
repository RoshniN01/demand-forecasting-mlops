import pandas as pd
import numpy as np
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
# 6. Train LightGBM
# --------------------------------------------------

print("\nTraining LightGBM model...")

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------

predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)


# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

# MAPE excluding zero actual values
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
print("FORECASTING RESULTS")
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