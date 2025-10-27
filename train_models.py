import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
os.makedirs('web_app', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load dataset
data = pd.read_csv('balanced_multiclass.csv')

# Define features and target
X = data.drop(columns=['label'])
y = data['label']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['proto', 'state'], drop_first=True)

# Replace '-' with NaN and handle missing values
X.replace('-', pd.NA, inplace=True)
X = X.apply(pd.to_numeric, errors='ignore')
X.fillna(X.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test indices for later use in Streamlit app
np.save('web_app/test_index.npy', X_test.index.values)

# -------------------------------------------------------------------------
# TRAINING MODELS
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

# ---------------- Random Forest ----------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# ---------------- Gradient Boosting ----------------
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
gb_acc = accuracy_score(y_test, y_pred_gb)
print(classification_report(y_test, y_pred_gb))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# ---------------- XGBoost ----------------
print("\nTraining XGBoost...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='mlogloss'
)
xgb.fit(X_train, y_train_enc)
y_pred_xgb_enc = xgb.predict(X_test)
y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print(classification_report(y_test, y_pred_xgb))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# ---------------- MLP (Neural Network) ----------------
print("\nTraining MLP...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(y_test, y_pred_mlp)
print(classification_report(y_test, y_pred_mlp))
print(f"MLP Accuracy: {mlp_acc:.4f}")

# ---------------- Logistic Regression ----------------
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    multi_class='multinomial', solver='lbfgs',
    max_iter=1000, random_state=42, C=1.0
)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)
print(classification_report(y_test, y_pred_lr))
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# -------------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Feature Importance Comparison
top_n = 15
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
rf_top = rf_importances.head(top_n)
axes[0, 0].barh(rf_top['Feature'], rf_top['Importance'], color='#1f77b4')
axes[0, 0].set_title('Random Forest - Top Features')
axes[0, 0].invert_yaxis()

gb_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False).head(top_n)
axes[0, 1].barh(gb_imp['Feature'], gb_imp['Importance'], color='#ff7f0e')
axes[0, 1].set_title('Gradient Boosting - Top Features')
axes[0, 1].invert_yaxis()

xgb_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False).head(top_n)
axes[1, 0].barh(xgb_imp['Feature'], xgb_imp['Importance'], color='#2ca02c')
axes[1, 0].set_title('XGBoost - Top Features')
axes[1, 0].invert_yaxis()

axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig('plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Model Accuracy Comparison
accuracies = {
    'Random Forest': rf_acc,
    'Gradient Boosting': gb_acc,
    'XGBoost': xgb_acc,
    'MLP': mlp_acc,
    'Logistic Regression': lr_acc
}
plt.figure(figsize=(10, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(),
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 1])
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}', ha='center', va='bottom')
plt.grid(axis='y', alpha=0.3)
plt.savefig('plots/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrices
models = [
    ('Random Forest', y_pred_rf),
    ('Gradient Boosting', y_pred_gb),
    ('XGBoost', y_pred_xgb),
    ('MLP', y_pred_mlp),
    ('Logistic Regression', y_pred_lr)
]
n_cols = 3
n_rows = (len(models) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
axes = np.atleast_2d(axes)
for i, (name, preds) in enumerate(models):
    r, c = divmod(i, n_cols)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=rf.classes_, yticklabels=rf.classes_, ax=axes[r, c])
    axes[r, c].set_title(f'{name} Confusion Matrix')
    axes[r, c].set_xlabel('Predicted')
    axes[r, c].set_ylabel('True')
for i in range(len(models), n_rows*n_cols):
    r, c = divmod(i, n_cols)
    axes[r, c].set_visible(False)
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Per-Class Precision, Recall, F1
metrics = []
for name, preds in models:
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average=None)
    for i, cls in enumerate(rf.classes_):
        metrics.append({'Model': name, 'Class': cls, 'Precision': p[i], 'Recall': r[i], 'F1-Score': f[i]})
metrics_df = pd.DataFrame(metrics)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, metric in enumerate(['Precision', 'Recall', 'F1-Score']):
    pivot = metrics_df.pivot(index='Class', columns='Model', values=metric)
    pivot.plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(metric)
    axes[idx].set_ylim([0, 1.1])
plt.tight_layout()
plt.savefig('plots/per_class_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Random Forest Learning Curve
def plot_learning_curve(est, X, y):
    from sklearn.model_selection import learning_curve
    sizes, train_scores, val_scores = learning_curve(est, X, y, cv=5, n_jobs=-1,
                                                     train_sizes=np.linspace(0.1, 1.0, 10))
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_scores.mean(axis=1), label="Train", color='blue')
    plt.plot(sizes, val_scores.mean(axis=1), label="CV", color='red')
    plt.legend()
    plt.title('Random Forest Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig('plots/learning_curve_random_forest.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_learning_curve(rf, X_train, y_train)

# -------------------------------------------------------------------------
# SAVE MODELS
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

joblib.dump(rf, 'web_app/random_forest_model.pkl')
joblib.dump(gb, 'web_app/gradient_boosting_model.pkl')
joblib.dump(xgb, 'web_app/xgb_model.pkl')
joblib.dump(le, 'web_app/label_encoder.pkl')
joblib.dump(mlp, 'web_app/mlp_model.pkl')
joblib.dump(lr, 'web_app/logistic_regression_model.pkl')
joblib.dump(scaler, 'web_app/scaler.pkl')

print("\nAll models saved successfully!")
print(f"\nAccuracies:")
print(f"  Random Forest: {rf_acc:.4f}")
print(f"  Gradient Boosting: {gb_acc:.4f}")
print(f"  XGBoost: {xgb_acc:.4f}")
print(f"  MLP: {mlp_acc:.4f}")
print(f"  Logistic Regression: {lr_acc:.4f}")
print("\nPlots saved in 'plots/' and models in 'web_app/'")
