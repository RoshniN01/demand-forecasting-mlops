from zenml import pipeline

from steps.feature_engineering import feature_engineering
from steps.train_model import train_model
from steps.forecast import forecast
from steps.evaluate import evaluate


@pipeline
def demand_forecasting_pipeline():

    # Step 1: Feature engineering
    ts = feature_engineering()

    # Step 2: Train model
    model = train_model(ts)

    # Step 3: Generate forecast
    predictions = forecast(ts, model)

    # Step 4: Evaluate forecast
    evaluate(predictions)