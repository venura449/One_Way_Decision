"""
Traffic Gate ML Training Script
================================
Trains two models from traffic_gate_dataset.csv:
  1. Classifier  → predict winner (A or B)
  2. Regressor   → predict gate_open_seconds

Usage:
  python train_traffic_gate.py
  python train_traffic_gate.py --data path/to/dataset.csv
  python train_traffic_gate.py --model xgboost   # or randomforest / both (default)
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "hour",
    "road_length_a_m", "road_length_b_m",
    "lane_a_car", "lane_a_motorcycle", "lane_a_bus", "lane_a_truck", "lane_a_total",
    "lane_b_car", "lane_b_motorcycle", "lane_b_bus", "lane_b_truck", "lane_b_total",
    # Engineered features (added below)
    "diff_total", "diff_weighted_score",
    "road_length_ratio", "rush_hour",
]

CLF_TARGET = "winner"
REG_TARGET = "gate_open_seconds"

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
WEIGHTS = {"car": 2.0, "motorcycle": 1.5, "bus": 3.0, "truck": 3.5}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Weighted traffic score per lane (mirrors gate time logic)
    df["score_a"] = (
        df["lane_a_car"] * WEIGHTS["car"] +
        df["lane_a_motorcycle"] * WEIGHTS["motorcycle"] +
        df["lane_a_bus"] * WEIGHTS["bus"] +
        df["lane_a_truck"] * WEIGHTS["truck"]
    )
    df["score_b"] = (
        df["lane_b_car"] * WEIGHTS["car"] +
        df["lane_b_motorcycle"] * WEIGHTS["motorcycle"] +
        df["lane_b_bus"] * WEIGHTS["bus"] +
        df["lane_b_truck"] * WEIGHTS["truck"]
    )

    # Key delta features the model can learn from
    df["diff_total"]          = df["lane_a_total"] - df["lane_b_total"]
    df["diff_weighted_score"] = df["score_a"] - df["score_b"]
    df["road_length_ratio"]   = df["road_length_a_m"] / (df["road_length_b_m"] + 1e-9)

    # Rush hour flag
    df["rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 19) else 0
    )

    return df


# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y_clf = (df[CLF_TARGET] == "B").astype(int)   # 0 = A, 1 = B
    y_reg = df[REG_TARGET]

    return X, y_clf, y_reg


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models(model_type: str):
    clf_models, reg_models = {}, {}

    if model_type in ("randomforest", "both"):
        clf_models["RandomForest"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=2,
                random_state=42, n_jobs=-1))
        ])
        reg_models["RandomForest"] = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=2,
                random_state=42, n_jobs=-1))
        ])

    if model_type in ("xgboost", "both"):
        if not XGBOOST_AVAILABLE:
            print("⚠  XGBoost not installed. Run: pip install xgboost")
        else:
            clf_models["XGBoost"] = XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0)

            reg_models["XGBoost"] = XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0)

    return clf_models, reg_models


# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X, y_clf, y_reg, clf_models, reg_models):
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    best_clf, best_reg = None, None
    best_clf_acc, best_reg_mae = 0, float("inf")

    print("\n" + "═" * 60)
    print("  CLASSIFIER  →  Predict winner (A or B)")
    print("═" * 60)

    for name, model in clf_models.items():
        model.fit(X_train, yc_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(yc_test, preds)
        cv    = cross_val_score(model, X, y_clf, cv=5, scoring="accuracy")

        print(f"\n▶ {name}")
        print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  CV  Accuracy  : {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(yc_test, preds, target_names=["Lane A", "Lane B"], digits=3))

        if acc > best_clf_acc:
            best_clf_acc = acc
            best_clf     = (name, model)

    print("\n" + "═" * 60)
    print("  REGRESSOR  →  Predict gate_open_seconds")
    print("═" * 60)

    for name, model in reg_models.items():
        model.fit(X_train, yr_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(yr_test, preds)
        r2    = r2_score(yr_test, preds)
        cv    = cross_val_score(model, X, y_reg, cv=5,
                                scoring="neg_mean_absolute_error")

        print(f"\n▶ {name}")
        print(f"  Test MAE  : {mae:.4f} seconds")
        print(f"  Test R²   : {r2:.4f}")
        print(f"  CV MAE    : {-cv.mean():.4f} ± {cv.std():.4f}")

        if mae < best_reg_mae:
            best_reg_mae = mae
            best_reg     = (name, model)

    return best_clf, best_reg


# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
def save_models(best_clf, best_reg):
    clf_path = os.path.join(OUTPUT_DIR, "winner_classifier.pkl")
    reg_path = os.path.join(OUTPUT_DIR, "gate_seconds_regressor.pkl")

    joblib.dump(best_clf[1], clf_path)
    joblib.dump(best_reg[1], reg_path)

    print("\n" + "═" * 60)
    print("  SAVED MODELS")
    print("═" * 60)
    print(f"  ✔ Classifier  [{best_clf[0]}]  →  {clf_path}")
    print(f"  ✔ Regressor   [{best_reg[0]}]  →  {reg_path}")
    return clf_path, reg_path


# ─────────────────────────────────────────────
# INFERENCE EXAMPLE
# ─────────────────────────────────────────────
def run_inference_example(clf_path: str, reg_path: str):
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)

    sample = pd.DataFrame([{
        "hour": 8,
        "road_length_a_m": 100,   "road_length_b_m": 200,
        "lane_a_car": 1,          "lane_a_motorcycle": 0,
        "lane_a_bus": 0,          "lane_a_truck": 0,   "lane_a_total": 1,
        "lane_b_car": 1,          "lane_b_motorcycle": 0,
        "lane_b_bus": 3,          "lane_b_truck": 0,   "lane_b_total": 4,
    }])
    sample = engineer_features(sample)
    sample = sample[FEATURE_COLS]

    winner      = "B" if clf.predict(sample)[0] == 1 else "A"
    gate_secs   = reg.predict(sample)[0]

    print("\n" + "═" * 60)
    print("  INFERENCE EXAMPLE  (matches your original JSON)")
    print("═" * 60)
    print(f"  Predicted winner       : Lane {winner}")
    print(f"  Predicted gate open    : {gate_secs:.2f} seconds")
    print("═" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Traffic Gate Model")
    parser.add_argument("--data",  default="traffic_gate_dataset.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--model", default="both",
                        choices=["randomforest", "xgboost", "both"],
                        help="Model type to train")
    args = parser.parse_args()

    print(f"\n📂 Loading data from: {args.data}")
    X, y_clf, y_reg = load_data(args.data)
    print(f"   Rows: {len(X)}  |  Features: {len(FEATURE_COLS)}")
    print(f"   Winner distribution: A={( y_clf==0).sum()}  B={(y_clf==1).sum()}")

    clf_models, reg_models = get_models(args.model)

    best_clf, best_reg = train_and_evaluate(X, y_clf, y_reg, clf_models, reg_models)

    clf_path, reg_path = save_models(best_clf, best_reg)

    run_inference_example(clf_path, reg_path)


if __name__ == "__main__":
    main()
