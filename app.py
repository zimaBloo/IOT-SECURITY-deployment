import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


# App config & const

st.set_page_config(page_title="IDS Mini-Project", layout="wide")

DATA_PATH= "balanced_multiclass.csv"
HER_MODELS_DIR= "web_app"
PLOTS_DIR  = "plots"
SEED= 42
TEST_SIZE= 0.2






#utilities

@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)



    df.columns = [c.strip() for c in df.columns]
    return df





def to_numeric_like_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match train_models.py numeric handling:
      - replace '-' with NaN
      - coerce numerics
      - fill NaN with each column's median (if all-NaN -> 0)
    """


    df = df.replace("-", np.nan)
    df = df.copy()
    num_cols = ["src_port","dst_port","duration","src_bytes","dst_bytes","src_pkts","dst_pkts"]
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            med = s.median()

            if pd.isna(med):


                med= 0
            df[c] = s.fillna(med)


    return df




def ensure_cat_for_her(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match training casing:
      - proto: lower-case (tcp/udp/icmp)
      - state: UPPERCASE (S0, SF, RSTO, RSTR, ...)
    """



    df = df.copy()
    if "proto" in df.columns:
        df["proto"] = df["proto"].fillna("unknown").astype(str).str.lower()
    else:
        df["proto"] = "unknown"



    if "state" in df.columns:
        df["state"] = df["state"].fillna("UNKNOWN").astype(str).str.upper()
    else:
        df["state"] = "UNKNOWN"



    return df





def encode_align_df_for_her(df_raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:



    """
    Align inputs to training space:
      - proto lower, state UPPER
      - numeric coercion + NaN -> column median
      - one-hot with drop_first=True
      - EXACT column order == feature_columns
    """



    df = df_raw.copy()
    df = ensure_cat_for_her(df)


    df = to_numeric_like_training(df)
    df_enc = pd.get_dummies(df, columns=["proto","state"], drop_first=True)







    # add missing expected cols
    for col in feature_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0
    # drop unexpected cols (only keep expected), preserve exact order


    df_enc = df_enc[feature_columns]
    return df_enc







def inverse_with_label_encoder(y_pred, label_encoder):
    """Use saved LabelEncoder (needed for XGBoost)."""
    if label_encoder is None:
        return y_pred
    try:




        return label_encoder.inverse_transform(np.asarray(y_pred).astype(int))
    except Exception:
        return y_pred


#load artifacts



def load_her_artifacts():
    arts = {"models": {}, "meta": {}}
    try:


        import joblib
    except Exception:


        st.error("joblib not installed — add it to requirements.txt")
        return arts
    







    le_path = os.path.join(HER_MODELS_DIR, "label_encoder.pkl")
    scaler_path = os.path.join(HER_MODELS_DIR, "scaler.pkl")



    label_encoder = joblib.load(le_path) if os.path.exists(le_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None




    arts["label_encoder"] = label_encoder
    arts["scaler"] = scaler 









    for name, fname, needs_scaler, uses_le in [


        ("Random Forest",       "random_forest_model.pkl",       False, False),
        ("Gradient Boosting",   "gradient_boosting_model.pkl",   False, False),
        ("XGBoost",             "xgb_model.pkl",                 False, True),


        ("MLP",                 "mlp_model.pkl",                 True,  False),
        ("Logistic Regression", "logistic_regression_model.pkl", True,  False),


    ]:
        


        path = os.path.join(HER_MODELS_DIR, fname)
        if not os.path.exists(path):
            continue


        try:
            mdl = joblib.load(path)
            arts["models"][name] = mdl



            feat_names = getattr(mdl, "feature_names_in_", None)
            if name in ("MLP", "Logistic Regression") and feat_names is None and scaler is not None:
                feat_names = getattr(scaler, "feature_names_in_", None)



            if feat_names is None:
                feat_names = []

            arts["meta"][name] = {
                "needs_scaler": needs_scaler,
                "uses_label_encoder": uses_le,
                "feature_names": list(feat_names)
            }


        except Exception as e:
            st.warning(f"Could not load {fname} from {HER_MODELS_DIR}: {e}")



    return arts







#load data and prep



if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Put your CSV there or update DATA_PATH.")
    st.stop()




df = load_df(DATA_PATH)
if "label" not in df.columns:
    st.error("Dataset must have a 'label' column.")
    st.stop()



df["label"] = df["label"].astype(str).str.strip()





arts= load_her_artifacts()
models= arts.get("models", {})
meta= arts.get("meta", {})
label_encoder= arts.get("label_encoder", None)
scaler  = arts.get("scaler", None)






TEST_INDEX_PATH = os.path.join(HER_MODELS_DIR, "test_index.npy")


saved_test_idx = np.load(TEST_INDEX_PATH, allow_pickle=True) if os.path.exists(TEST_INDEX_PATH) else None




#UI Tabs!



tab_overview, tab_results, tab_importance, tab_predict = st.tabs(



    ["Overview", "Results", "Feature insights", "Prediction demo"]

)

#overview


with tab_overview:
    st.title("Intrusion Detection — Project Overview")


    st.markdown("• We classify IoT-23 flows into **Benign**, **C&C**, **DDoS**, **PortScan**, and **Okiru**.")
    st.markdown("• Models: Random Forest, Gradient Boosting, XGBoost, MLP, and Logistic Regression.")



    st.markdown("• This page evaluates the models and lets you try single predictions.")




    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(12), use_container_width=True)




    with c2:
        st.subheader("Shape")
        st.write(df.shape)
        st.subheader("Columns")
        st.write(list(df.columns))





    st.subheader("Class balance")
    sns.set(style="whitegrid")
    order = df["label"].value_counts().index

    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="label", palette="Set2", order=order, ax=ax)



    ax.set_title("Label Distribution", fontsize=14)
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)


    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True)








#Results


with tab_results:
    st.header("Results")



    st.markdown("- We evaluate each loaded model on a test set.")




    rows = []
    cm_cache = {}




    announced_saved_split = False


    if models:

        y_all = df["label"].astype(str).values

        for name, mdl in models.items():


            info = meta.get(name, {})
            needs_scaler = info.get("needs_scaler", False)
            uses_le= info.get("uses_label_encoder", False)





            expected_cols = info.get("feature_names", [])



            if not expected_cols:

                tmp = ensure_cat_for_her(df.drop(columns=["label"]).copy())
                tmp = to_numeric_like_training(tmp)



                tmp = pd.get_dummies(tmp, columns=["proto","state"], drop_first=True)
                expected_cols = list(tmp.columns)





            X_all = encode_align_df_for_her(df.drop(columns=["label"]), expected_cols)





            if saved_test_idx is not None:
                if not announced_saved_split:



                    st.caption("Using the original training-time test split")
                    announced_saved_split = True



                X_te= X_all.iloc[saved_test_idx]
                y_te= y_all[saved_test_idx]



            else:



                _, X_te, _, y_te = train_test_split(
                    X_all, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
                )




            #
            if needs_scaler and scaler is not None:
                X_te = pd.DataFrame(scaler.transform(X_te), columns=expected_cols)





            try:
                y_pred = mdl.predict(X_te)
            except Exception as e:
                st.warning(f"{name} failed to predict: {e}")
                continue





            y_pred_str = inverse_with_label_encoder(y_pred, label_encoder) if uses_le else y_pred




            acc = accuracy_score(y_te, y_pred_str)
            rows.append({"Model": name, "Accuracy": acc})




            labels_sorted = sorted(np.unique(np.concatenate([y_te, y_pred_str])))
            cm_cache[name] = (y_te, y_pred_str, labels_sorted)




        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Accuracy",     ascending=False), use_container_width=True)



    else:   


        st.info(f"No models found in '{HER_MODELS_DIR}/'. Place the .pkl files there and refresh.")




    st.subheader("Training Plots")



    image_paths = sorted(
        glob.glob(os.path.join(PLOTS_DIR, "*.png")) +
        glob.glob(os.path.join(PLOTS_DIR, "*.jpg"))
    )




    if not image_paths:
        st.info(f"No images found in '{PLOTS_DIR}/'.")
    else:
        cols_per_row = 3
        for i in range(0, len(image_paths), cols_per_row):

            cols = st.columns(cols_per_row)
            for col, path in zip(cols, image_paths[i:i+cols_per_row]):
                col.image(path, use_container_width=True, caption=os.path.basename(path))







    st.subheader("Confusion Matrices")
    if cm_cache:



        model_to_show = st.selectbox(
            "Select a model to visualize confusion matrix",
            list(cm_cache.keys()),

            key="cm_model_selectbox"
        )


        y_te, y_pred_str, labels_sorted = cm_cache[model_to_show]
        cm = confusion_matrix(y_te, y_pred_str, labels=labels_sorted)
        fig, ax = plt.subplots(figsize=(4.6, 3.8))



        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_to_show} — Confusion Matrix", fontsize=11)



        ax.tick_params(axis="both", labelsize=9)
        texts = disp.text_.ravel() if isinstance(disp.text_, np.ndarray) else [disp.text_]




        for t in texts:
            if hasattr(t, "set_fontsize"):
                t.set_fontsize(8)


        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# Feature insights


with tab_importance:



    st.header("Feature insights")
    if models:
        model_name = st.selectbox(
            "Model",
            list(models.keys()),
            key="fi_model_selectbox"
        )




        info = meta.get(model_name, {})
        expected_cols = info.get("feature_names", [])
        if not expected_cols:



            tmp = ensure_cat_for_her(df.drop(columns=["label"]).copy())
            tmp = to_numeric_like_training(tmp)


            tmp = pd.get_dummies(tmp, columns=["proto","state"], drop_first=True)
            expected_cols = list(tmp.columns)

        top_n = st.slider("Top N features", 5, 30, 10, 1, key="fi_topn_slider")




        def compute_feature_importance(model, cols):
            if hasattr(model, "coef_"):
                coefs = model.coef_





                
                imp = np.mean(np.abs(coefs), axis=0) if coefs.ndim == 2 else np.abs(coefs)
                return pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
            


            if hasattr(model, "feature_importances_"):

                imp = model.feature_importances_
                return pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False)
            

            return None




        fi = compute_feature_importance(models[model_name], expected_cols)


        if fi is None:

            st.info("This model does not expose standard feature importances.")





        else:
            top = fi.head(top_n)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.barh(top["feature"][::-1], top["importance"][::-1])



            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} features — {model_name}")
            st.pyplot(fig, clear_figure=True)



            st.dataframe(top.reset_index(drop=True), use_container_width=True)
            
    else:

        st.info(f"Place model files in '{HER_MODELS_DIR}/' to enable feature insights.")







#predictions!



with tab_predict:
    st.header("Try a single prediction")
    st.caption("Enter a sample network flow.")




    if not models:
        st.error("No models loaded.")
        st.stop()






    model_choice = st.selectbox(
        "Choose model",
        list(models.keys()),
        index=0,

        key="pred_model_selectbox"



    )




    col1, col2, col3 = st.columns(3)




    with col1:
        src_port = st.number_input("src_port", min_value=0, value=41161, key="pred_src_port")
        dst_port = st.number_input("dst_port", min_value=0, value=23, key="pred_dst_port")
        duration = st.number_input("duration", min_value=0.0, value=0.0, step=0.001, format="%.3f", key="pred_duration")


    with col2:
        src_bytes = st.number_input("src_bytes", min_value=0, value=0, key="pred_src_bytes")
        dst_bytes = st.number_input("dst_bytes", min_value=0, value=0, key="pred_dst_bytes")
        src_pkts  = st.number_input("src_pkts",  min_value=0, value=1, key="pred_src_pkts")


    with col3:
        dst_pkts  = st.number_input("dst_pkts",  min_value=0, value=0, key="pred_dst_pkts")
        proto     = st.selectbox("proto", ["tcp","udp","icmp","unknown"], index=0, key="pred_proto")
        state     = st.selectbox("state", ["S0","SF","CON","FIN","RST","RSTO","RSTR","S1","S3","UNKNOWN"], index=0, key="pred_state")





    if st.button("🔍 Run Prediction", key="pred_button"):
        mdl = models[model_choice]
        info = meta.get(model_choice, {})


        needs_scaler = info.get("needs_scaler", False)
        expected_cols = info.get("feature_names", [])




        if not expected_cols:
            tmp = ensure_cat_for_her(df.drop(columns=["label"]).copy())
            tmp = to_numeric_like_training(tmp)

            tmp = pd.get_dummies(tmp, columns=["proto","state"], drop_first=True)
            expected_cols = list(tmp.columns)




        sample = pd.DataFrame([{

            "src_port": src_port, "dst_port": dst_port, "duration": duration,
            "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "src_pkts": src_pkts, "dst_pkts": dst_pkts,


            "proto": proto.lower(), "state": state.upper()
        }])





        X_one = encode_align_df_for_her(sample, expected_cols)



        if needs_scaler and scaler is not None:
            X_one = pd.DataFrame(scaler.transform(X_one), columns=expected_cols)


        try:
            yhat = mdl.predict(X_one)
            uses_le = info.get("uses_label_encoder", False)
            pred = inverse_with_label_encoder(yhat, label_encoder) if uses_le else yhat




            conf_txt = ""


            if hasattr(mdl, "predict_proba"):



                try:
                    proba = mdl.predict_proba(X_one)[0]
                    top_idx = int(np.argmax(proba))
                    top_p = float(proba[top_idx])
                    conf_txt = f" (confidence {top_p:.2%})"


                except Exception:
                    pass



                




                # green for benign
            pred_label = str(pred[0])
            if "benign" in pred_label.lower():
                st.success(f"✅ Benign traffic detected{conf_txt}")




                #red for attacks
            else:
                st.error(f"🚨 Potential **{pred_label}** attack detected{conf_txt}")
            st.caption(f"Model used: {model_choice}")



        except Exception as e:
            st.error(f"Prediction failed: {e}")
