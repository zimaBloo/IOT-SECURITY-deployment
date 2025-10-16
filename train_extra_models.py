# train_extra_models.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_extra_models(csv_file, models_dir="models"):
    print(f"Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")

    # Encode labels
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    df["label_enc"], uniques = pd.factorize(df["label"])

    # Clean and convert numeric columns
    numeric_cols = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
    df = df.replace("-", np.nan)  # changed: normalize placeholder dashes to NaN

    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")  # changed: coerce column to numeric
        if s.notna().any():
            df[c] = s.fillna(s.mean())             # changed: compute mean on clean numeric
        else:
            df[c] = s.fillna(0)                    # changed: handle all-NaN edge case

    # One-hot encode categorical columns
    cat_cols = ["proto", "state"]
    df = pd.get_dummies(df, columns=cat_cols)

    # Split features/target
    X = df.drop(['label', 'label_enc'], axis=1)
    y = df['label_enc']

    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    results["Random Forest"] = rf_acc
    joblib.dump(rf, os.path.join(models_dir, "random_forest_model.pkl"))

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    xgb.fit(X_train, y_train)
    xgb_acc = xgb.score(X_test, y_test)
    results["XGBoost"] = xgb_acc
    joblib.dump(xgb, os.path.join(models_dir, "xgboost_model.pkl"))

    # --- CatBoost ---
    print("\nTraining CatBoost...")
    cat = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_state=42,
        verbose=200
    )
    cat.fit(X_train, y_train)
    cat_acc = cat.score(X_test, y_test)
    results["CatBoost"] = cat_acc
    joblib.dump(cat, os.path.join(models_dir, "catboost_model.pkl"))

    # Save mappings and feature info
    label_mapping = {idx: label for idx, label in enumerate(uniques)}
    joblib.dump(label_mapping, os.path.join(models_dir, "label_mapping.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl"))

    print("\n✅ Training complete. Model accuracies:")
    for model, acc in results.items():
        print(f"  - {model}: {acc:.4f}")

    return results


if __name__ == "__main__":
    results = train_extra_models(os.path.join("data", "balanced_multiclass.csv"))
