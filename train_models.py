import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

def train_models(csv_file):
    print(f"Loading preprocessed data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows")

    # CLEANING
    df = df.replace("-", np.nan)

    num_cols = ["src_port","dst_port","duration","src_bytes","dst_bytes","src_pkts","dst_pkts"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ["proto","state"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("unknown")

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["label"] != "FileDownload"]  # DROP FileDownload entirely

        print("\nLabel distribution:")
        print(df["label"].value_counts())

        df["label_enc"], uniques = pd.factorize(df["label"])
        print("\nLabel encoding mapping:")
        for idx, label in enumerate(uniques):
            print(f"{label}: {idx}")
    else:
        raise ValueError("Expected a 'label' column in the dataset.")

    cat_cols = ["proto", "state"]
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])

    X = df.drop(['label', 'label_enc'], axis=1)
    y = df['label_enc']

    print(f"\nFeature columns ({len(X.columns)}): {X.columns.tolist()}")
    print(f"Dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    lr_accuracy = lr.score(X_test, y_test)
    print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")

    if lr.coef_.shape[0] > 1:
        mean_coef = np.mean(np.abs(lr.coef_), axis=0)
        importances_lr = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_coefficient': mean_coef
        }).sort_values('mean_abs_coefficient', ascending=False)
        print("\nTop 10 Logistic Regression feature importances (mean abs coef):")
        print(importances_lr.head(10))
    else:
        importances_lr = pd.DataFrame({
            'feature': X.columns,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=np.abs, ascending=False)
        print("\nTop 10 Logistic Regression feature importances:")
        print(importances_lr.head(10))

    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_accuracy = dt.score(X_test, y_test)
    print(f"Decision Tree accuracy: {dt_accuracy:.4f}")

    importances_dt = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Decision Tree feature importances:")
    print(importances_dt.head(10))

    plt.figure(figsize=(16, 8))
    plot_tree(
        dt,
        feature_names=X.columns,
        class_names=[str(c) for c in sorted(y.unique())],
        filled=True,
        max_depth=3
    )
    plt.title("Decision Tree Visualization (max_depth=3)")
    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaving trained models and artifacts...")
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    label_mapping = {idx: label for idx, label in enumerate(uniques)}

    joblib.dump(lr, os.path.join(out_dir, 'logistic_regression_model.pkl'))
    joblib.dump(dt, os.path.join(out_dir, 'decision_tree_model.pkl'))
    joblib.dump(label_mapping, os.path.join(out_dir, 'label_mapping.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(out_dir, 'feature_columns.pkl'))

    print("✓ Saved models/logistic_regression_model.pkl")
    print("✓ Saved models/decision_tree_model.pkl")
    print("✓ Saved models/label_mapping.pkl")
    print("✓ Saved models/feature_columns.pkl")
    print("✓ Saved decision_tree_visualization.png")

    return lr, dt, X_test, y_test


if __name__ == "__main__":
    data_path = os.path.join("data", "balanced_multiclass.csv")
    lr_model, dt_model, X_test, y_test = train_models(data_path)
