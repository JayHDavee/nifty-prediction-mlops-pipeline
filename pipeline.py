import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------
# CONFIG
# -----------------------------

EXPERIMENT_NAME = "Nifty_Direction_Prediction"

os.makedirs("models", exist_ok=True)


# -----------------------------
# DATA INGESTION
# -----------------------------

print("Downloading NIFTY data...")

df = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2026-01-01"
)

df.columns = df.columns.get_level_values(0)


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

print("Creating features...")

df["Return_1d"] = df["Close"].pct_change(1)
df["Return_3d"] = df["Close"].pct_change(3)
df["Return_5d"] = df["Close"].pct_change(5)

df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

df["Volatility_5"] = df["Return_1d"].rolling(5).std()
df["Volatility_10"] = df["Return_1d"].rolling(10).std()

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

df = df.dropna()


# -----------------------------
# FEATURES
# -----------------------------

features = [
    "Return_1d",
    "Return_3d",
    "Return_5d",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility_5",
    "Volatility_10"
]

X = df[features]
y = df["Target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# MLFLOW
# -----------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)


# -----------------------------
# MODELS
# -----------------------------

models = {
    "LogisticRegression":
        LogisticRegression(max_iter=1000),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
}


best_model = None
best_model_name = None
best_accuracy = 0


# -----------------------------
# TRAINING LOOP
# -----------------------------

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name
        )

        print(
            f"{model_name}: "
            f"Accuracy={accuracy:.4f}"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name


# -----------------------------
# SAVE BEST MODEL
# -----------------------------

best_model_path = "models/nifty_best_model.pkl"

joblib.dump(
    best_model,
    best_model_path
)

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")
print(f"Saved To   : {best_model_path}")
