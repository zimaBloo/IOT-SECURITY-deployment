import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


st.set_page_config(page_title="IDS Mini-Project", layout="wide")

DATA_PATH = "data/balanced_multiclass.csv"
MODELS_DIR = "models"
FIGURES_DIR = "figures"
SEED = 42
TEST_SIZE = 0.2

#utils
def load_df(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def to_numeric(df):
    for c in ["src_port","dst_port","duration","src_bytes","dst_bytes","src_pkts","dst_pkts"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce").fillna(0)
    return df

def ensure_cat(df):
    for c in ["proto","state"]:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str).str.lower()
    return df

def encode_align(df_raw, feature_columns):
    df = df_raw.copy()
    df = to_numeric(df)
    df = ensure_cat(df)
    df_enc = pd.get_dummies(df, columns=["proto","state"])
    # add missing expected cols
    for col in feature_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0
    # keep only expected
    df_enc = df_enc[feature_columns]
    return df_enc

def load_artifacts():
    arts = {"models": {}}
    try:
        import joblib
    except Exception:
        st.warning("joblib not installed — add it to requirements.txt")
        return arts
    # features+labels
    fc = os.path.join(MODELS_DIR, "feature_columns.pkl")
    lm = os.path.join(MODELS_DIR, "label_mapping.pkl")
    if os.path.exists(fc):
        arts["feature_columns"] = joblib.load(fc)
    if os.path.exists(lm):
        arts["label_mapping"] = joblib.load(lm)  # idx -> label
    # models (load if present)
    for name, fname in [
        ("Logistic Regression", "logistic_regression_model.pkl"),
        ("Decision Tree",       "decision_tree_model.pkl"),
        ("Random Forest",       "random_forest_model.pkl"),
        ("XGBoost",             "xgboost_model.pkl"),
        ("CatBoost",            "catboost_model.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                arts["models"][name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")
    return arts

def map_indices(y_pred, label_mapping):
    if not isinstance(label_mapping, dict):
        return y_pred
    return [label_mapping.get(int(i), str(i)) for i in y_pred]

def feature_importance(model, cols):
    if hasattr(model, "coef_"):
        coefs = model.coef_
        imp = np.mean(np.abs(coefs), axis=0) if coefs.ndim == 2 else np.abs(coefs)
        return pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
    return None

#load data/artifacts
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Put your CSV there.")
    st.stop()
df = load_df(DATA_PATH)
if "label" not in df.columns:
    st.error("Dataset must have a 'label' column.")
    st.stop()

# --- DROP FileDownload rows for the app views/eval ---
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"] != "FileDownload"]  # removed singleton class

arts = load_artifacts()
feature_columns = arts.get("feature_columns")
models = arts.get("models", {})
label_mapping = arts.get("label_mapping")

#tabs
tab_overview, tab_results, tab_importance, tab_predict = st.tabs(
    ["Overview", "Results", "Feature insights", "Prediction demo"]
)

# ---------- Overview ----------
with tab_overview:
    st.title("Intrusion Detection — Project Overview")
    st.markdown("•   Today, we’re surrounded by smart devices from cameras to coffee machines and all are connected to the internet.")
    st.markdown("•   But that also means there are more ways for hackers to attack.")
    st.markdown("•   Our project explores how AI can automatically spot suspicious network activity before it becomes a problem.")
    st.markdown("•   We cleaned real IoT traffic data and trained models to tell normal behavior from potential attacks.")
    st.markdown("•   Then we compared how well different approaches performed and visualized the results.")
    st.markdown("•   Finally, we built a simple demo where anyone can try it out and see how the system reacts in real time.")

    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(12), width="stretch")
    with c2:
        st.subheader("Shape")
        st.write(df.shape)
        st.subheader("Columns")
        st.write(list(df.columns))

    st.subheader("Class balance")

    # same formatting as your Colab snippet
    sns.set(style="whitegrid")
    order = df["label"].astype(str).str.strip().value_counts().index

    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="label", palette="Set2", order=order, ax=ax)

    ax.set_title("Label Distribution", fontsize=14)
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45)

    st.pyplot(fig, clear_figure=True)


# Results
with tab_results:
    st.header("Results")
    st.markdown(
        "- We trained multiple models (e.g., Logistic Regression, Decision Tree).  \n"
        "- Below we show quick accuracy on a small holdout from this dataset."
    )

    if feature_columns and models:
        # prepare X/y
        X = encode_align(df, feature_columns)
        y = df["label"].values
        # if labels are strings and models were trained on integers, invert map
        inv = {v: k for k, v in label_mapping.items()} if isinstance(label_mapping, dict) else None
        if inv:
            y_enc = np.array([inv.get(str(v), np.nan) for v in y])
            mask = ~pd.isna(y_enc)
            X_eval = X[mask]
            y_eval = y_enc[mask].astype(int)
        else:
            X_eval, y_eval = X, y

        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = None, None, None, None

        # keep only classes with at least 2 samples (safety)
        vals, counts_arr = np.unique(y_eval, return_counts=True)
        valid_classes = {int(v) for v, c in zip(vals, counts_arr) if c >= 2}
        if len(valid_classes) < len(vals):
            dropped_labels = [label_mapping.get(int(v), str(v)) for v, c in zip(vals, counts_arr) if c < 2]
            st.warning("Dropped classes with < 2 samples for evaluation: " + ", ".join(map(str, dropped_labels)))
            keep_mask = np.array([int(y) in valid_classes for y in y_eval])
            X_eval = X_eval[keep_mask]
            y_eval = y_eval[keep_mask]

        if len(np.unique(y_eval)) >= 2:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_eval, y_eval, test_size=0.2, random_state=SEED, stratify=y_eval
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_eval, y_eval, test_size=0.2, random_state=SEED, stratify=None
            )

        rows = []
        for name, mdl in models.items():
            try:
                acc = mdl.score(X_te, y_te)
                rows.append({"Model": name, "Accuracy": acc})
            except Exception as e:
                rows.append({"Model": name, "Accuracy": np.nan})
                st.warning(f"{name} failed to score: {e}")
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Accuracy", ascending=False), use_container_width=True)
            




    st.subheader("Confusion Matrices")

    model_to_show = st.selectbox(
        "Select a model to visualize confusion matrix",
        list(models.keys())
    )

    mdl = models[model_to_show]
    try:
        # classes present in the current test set only
        present_classes = np.array(sorted(np.unique(y_te), key=int))
        display_labels = [label_mapping.get(int(c), str(c)) for c in present_classes]

        y_pred = mdl.predict(X_te)
        cm = confusion_matrix(y_te, y_pred, labels=present_classes)

        col_plot, _ = st.columns([0.55, 0.45])
        with col_plot:
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            disp.plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{model_to_show} — Confusion Matrix", fontsize=11)
            ax.tick_params(axis="both", labelsize=9)

            # shrink text font size (handles both 1D and 2D text_ arrays)
            texts = disp.text_.ravel() if isinstance(disp.text_, np.ndarray) else disp.text_
            for t in texts:
                t.set_fontsize(8)

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.warning(f"Could not compute confusion matrix for {model_to_show}: {e}")






    st.subheader("Visualisations")

    import glob

    # Find both .png and .jpg images
    image_paths = sorted(
        glob.glob(os.path.join(FIGURES_DIR, "*.png")) +
        glob.glob(os.path.join(FIGURES_DIR, "*.jpg"))
    )

    if not image_paths:
        st.info(f"No images found in '{FIGURES_DIR}/'.")
    else:
        cols_per_row = 3  # Number of images per row
        for i in range(0, len(image_paths), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, path in zip(cols, image_paths[i:i+cols_per_row]):
                col.image(path, use_container_width=True)





# Feature insights
with tab_importance:
    st.header("Feature insights")
    if feature_columns and models:
        model_name = st.selectbox("Choose a model", list(models.keys()))
        top_n = st.slider("Top N features", 5, 30, 10, 1)
        fi = feature_importance(models[model_name], feature_columns)
        if fi is None:
            st.info("This model does not expose standard feature importances.")
        else:
            top = fi.head(top_n)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(top["feature"][::-1], top["importance"][::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} features — {model_name}")
            st.pyplot(fig, clear_figure=True)
            st.dataframe(top.reset_index(drop=True))
    else:
        st.info("Place model files in models/ to enable feature importance.")

#Prediction demo
with tab_predict:
    st.header("Try a single prediction")
    st.caption("Enter a sample network flow.")

    if models:
        model_choice = st.selectbox(
            "Choose model for prediction",
            list(models.keys()),
            index=0
        )
    else:
        model_choice = None

    col1, col2, col3 = st.columns(3)
    with col1:
        src_port = st.number_input("src_port", min_value=0, value=41161)
        dst_port = st.number_input("dst_port", min_value=0, value=23)
        duration = st.number_input("duration", min_value=0.0, value=0.0, step=0.001, format="%.3f")
    with col2:
        src_bytes = st.number_input("src_bytes", min_value=0, value=0)
        dst_bytes = st.number_input("dst_bytes", min_value=0, value=0)
        src_pkts  = st.number_input("src_pkts",  min_value=0, value=1)
    with col3:
        dst_pkts  = st.number_input("dst_pkts",  min_value=0, value=0)
        proto     = st.selectbox("proto", ["tcp","udp","icmp","unknown"], index=0)
        state     = st.selectbox("state", ["s0","sf","con","fin","rst","unknown"], index=0)

    if st.button("🔍 Run Prediction"):
        if not feature_columns or not models or not model_choice:
            st.error("Missing model files or feature_columns.pkl in models/.")
        else:
            sample = pd.DataFrame([{
                "src_port": src_port, "dst_port": dst_port, "duration": duration,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "src_pkts": src_pkts, "dst_pkts": dst_pkts,
                "proto": proto, "state": state
            }])

            X_one = encode_align(sample, feature_columns)
            mdl = models[model_choice]
            try:
                yhat = mdl.predict(X_one)
                pred = map_indices(yhat, label_mapping)
                conf_txt = ""
                if hasattr(mdl, "predict_proba"):
                    proba = mdl.predict_proba(X_one)[0]
                    top_p = float(proba[yhat[0]])
                    conf_txt = f" (confidence {top_p:.2%})"

                pred_label = str(pred[0]).lower()
                if "benign" in pred_label:
                    st.success(f"✅ Benign traffic detected{conf_txt}")
                else:
                    st.error(f"🚨 Potential **{pred[0]}** attack detected{conf_txt}")
                st.caption(f"Model used: {model_choice}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
