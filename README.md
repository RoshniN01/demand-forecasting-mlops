# Demand Forecasting MLOps

## Problem Statement

Demand forecasting is an important task in retail and e-commerce because accurate demand predictions can help businesses improve inventory planning  and support better decision-making.

This project focuses on building a machine learning-based demand forecasting system using the M5 retail sales dataset. The project applies MLOps practices to make the forecasting workflow reproducible, track experiments, optimize model hyperparameters, and organize the complete machine learning workflow.

## Objective

The main objective of this project is to develop a demand forecasting pipeline using the M5 retail sales dataset and apply MLOps practices for reproducible model development and experiment tracking.

The project aims to:

- Perform time-series feature engineering using lag and rolling features.
- Train a LightGBM gradient-boosting model for demand forecasting.
- Optimize model hyperparameters using Optuna.
- Track model parameters and evaluation metrics such as RMSE and MAPE using MLflow.
- Orchestrate the workflow using ZenML.
- Maintain the project code and documentation using GitHub.

## Dataset

This project uses the **M5 Forecasting — Accuracy** dataset from Kaggle, which contains hierarchical daily sales data from Walmart.

The following three datasets are used:

- `calendar.csv` — contains calendar and event-related information.
- `sales_train_evaluation.csv` — contains historical daily sales data for products across stores.
- `sell_prices.csv` — contains product price information across stores and weeks.

All three datasets are stored in the `data/` directory.

## Tools Used

This project uses the following MLOps tools:

| Tool | Purpose |
|------|---------|
| **ZenML** | Orchestrates the demand forecasting pipeline |
| **Optuna** | Performs hyperparameter optimization for the LightGBM model |
| **MLflow** | Tracks experiments, parameters, and evaluation metrics such as RMSE and MAPE |
| **GitHub** | Provides version control and hosts the project repository |

## Project Architecture

The demand forecasting workflow is organized as a ZenML pipeline. The pipeline performs feature engineering, model training, forecasting, and evaluation.

### Pipeline Flow

M5 Dataset
↓
Feature Engineering
(Lag & Rolling Features)
↓
Optuna Hyperparameter Optimization
↓
LightGBM Model Training
↓
Demand Forecasting
↓
Model Evaluation
(RMSE & MAPE)
↓
MLflow Experiment Tracking

ZenML → Pipeline Orchestration
GitHub → Version Control & Project Repository

## Project Structure

```text
demand-forecasting-mlops/
│
├── data/
│   ├── calendar.csv
│   ├── sales_train_evaluation.csv
│   └── sell_prices.csv
│
├── pipelines/
│   └── forecast_pipeline.py
│
├── src/
│   ├── train_forecast.py
│   └── train_forecast_backup.py
│
├── steps/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── forecast.py
│   └── evaluate.py
│
├── .github/
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
└── run_pipeline.py

## Feature Engineering

The feature engineering stage prepares the historical sales data for demand forecasting.

The pipeline creates time-series features that help the model learn patterns from previous demand observations.

The main features include:

- **Lag features** — use previous sales observations as input features.
- **Rolling features** — calculate statistics over previous observations to capture demand trends.
- **Calendar-based features** — use information from the calendar dataset to capture time-related patterns.

These engineered features are then passed to the LightGBM model for training and forecasting.

## Model

The project uses **LightGBM (Light Gradient Boosting Machine)** as the demand forecasting model.

LightGBM is a gradient-boosting algorithm suitable for structured and tabular data. It is trained using the engineered time-series features generated from the M5 dataset.

The model is evaluated using forecasting metrics such as:

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

## Hyperparameter Optimization with Optuna

Optuna is used to optimize the hyperparameters of the LightGBM model.

Optuna automatically explores different combinations of hyperparameters and evaluates their performance to identify a better configuration for the forecasting model.

The optimization process follows these steps:

1. Define the hyperparameter search space.
2. Generate a trial with a set of hyperparameters.
3. Train the LightGBM model using the selected parameters.
4. Evaluate the model using the selected evaluation metric.
5. Repeat the process for multiple trials.
6. Select the best-performing hyperparameter configuration.

The best parameters obtained through Optuna are used for the final model training and forecasting.

## MLflow Experiment Tracking

MLflow is used to track the experiments performed during model development.

For each experiment run, MLflow can be used to record:

- Model parameters
- Hyperparameters
- RMSE
- MAPE
- Model performance information

This allows different experiments and model configurations to be compared and helps identify the best-performing model.

### MLflow Workflow

Model Training
↓
MLflow Experiment Run
↓
Log Parameters
↓
Log RMSE and MAPE
↓
Compare Experiments

### MLflow Dashboard

The MLflow dashboard screenshot will be added here to show the tracked experiments and evaluation metrics.

## ZenML Pipeline

ZenML is used to orchestrate the demand forecasting workflow as a reproducible machine learning pipeline.

The pipeline consists of the following stages:

1. **Feature Engineering** — creates the required lag, rolling, and other forecasting features.
2. **Model Training** — trains the LightGBM model using the engineered features.
3. **Forecasting** — generates demand forecasts using the trained model.
4. **Evaluation** — evaluates the forecasts using metrics such as RMSE and MAPE.

### Pipeline Flow

Feature Engineering
↓
Model Training
↓
Forecasting
↓
Evaluation

ZenML manages the execution of these pipeline steps and provides a structured workflow for reproducing the forecasting process.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RoshniN01/demand-forecasting-mlops.git
cd demand-forecasting-mlops
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Command Prompt:

```cmd
venv\Scripts\activate
```

For PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

After installing the required dependencies, run the main pipeline using:

```bash
python run_pipeline.py
```

The pipeline executes the demand forecasting workflow, including:

1. Feature engineering
2. Model training
3. Forecast generation
4. Model evaluation

## Reproducing Results

To reproduce the results:

1. Clone the GitHub repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies using `requirements.txt`.
4. Ensure the M5 datasets are available in the `data/` directory.
5. Run the pipeline using `python run_pipeline.py`.
6. View the ZenML pipeline execution.
7. View the experiment runs and metrics using MLflow.

## Results & Metrics

The forecasting model was evaluated using the following metrics:

| Metric | Value |
|--------|-------|
| RMSE | 1.3836 |
| MAPE | 42.84% |

These metrics were obtained during the model evaluation stage of the forecasting pipeline.

The results can be further compared across different Optuna trials and MLflow experiment runs.

## MLflow Experiment Tracking

MLflow was used to track the experiments, model parameters, and evaluation metrics such as RMSE and MAPE.

### MLflow Dashboard

The following screenshot shows the MLflow experiment tracking interface used in the project.

![MLflow Dashboard](images/mlflow.jpeg)

### ZenML Pipeline Execution

The following screenshot shows the ZenML pipeline execution.

![ZenML Pipeline](images/zenml.jpeg)

## Architecture / Pipeline

The demand forecasting system follows a sequential machine learning pipeline orchestrated using ZenML.

```text
M5 Retail Sales Dataset
          │
          ▼
   Data Preparation
          │
          ▼
  Feature Engineering
 (Lags, Rolling Statistics,
  Day/Month Features)
          │
          ▼
     LightGBM Model
          │
          ▼
      Forecasting
          │
          ▼
       Evaluation
    (RMSE and MAPE)
          │
          ▼
    MLflow Tracking


## Conclusion

This project demonstrates an end-to-end demand forecasting workflow using the M5 retail sales dataset.
The project combines machine learning with MLOps practices by using LightGBM for demand forecasting, Optuna for hyperparameter optimization, ZenML for pipeline orchestration, MLflow for experiment tracking, and GitHub for version control and project collaboration.
The workflow provides a structured and reproducible approach to developing, optimizing, tracking, and evaluating a demand forecasting model.

## Team Members

- Roshni N
- Sanapathi Veda Charan
- Vinay V