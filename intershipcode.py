import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# Streamlit config
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Online Payment Fraud Detection with XGBoost")

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("## 📘 About this App")
    st.write("""
    This app demonstrates **Online Payment Fraud Detection** using **XGBoost**.

    🔹 Handles data preprocessing (encoding, scaling, SMOTE balancing).  
    🔹 Trains and evaluates an XGBoost model.  
    🔹 Allows uploading a test dataset for predictions.  
    🔹 Provides metrics, confusion matrix, and feature importance.  
    """)

# --- TRAINING SECTION ---
st.header("1️⃣ Upload Training Dataset")

train_file = st.file_uploader("Upload training dataset (CSV)", type=["csv"], key="train")

if train_file is not None:
    train_df = pd.read_csv(train_file)
    st.success(f"✅ Training data uploaded. Total rows: {len(train_df)}")

    st.subheader("📄 Full Training Dataset")
    st.dataframe(train_df, use_container_width=True)

    if "isFraud" not in train_df.columns:
        st.error("❌ Training data must include 'isFraud' column!")
    else:
        if st.button("🚀 Train Model"):
            with st.spinner("Training the model..."):

                # ---- Encoding ----
                type_dummies = pd.get_dummies(train_df['type'], drop_first=True)
                train_df = pd.concat([train_df, type_dummies], axis=1)

                # ---- Features and Target ----
                X = train_df.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
                y = train_df['isFraud']

                # ---- Scaling ----
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # ---- Train/Test Split ----
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.3, stratify=y, random_state=42
                )

                # ---- SMOTE ----
                sm = SMOTE(random_state=42)
                X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

                # ---- Model Training ----
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
                model.fit(X_train_res, y_train_res)

                # ---- Evaluation ----
                val_preds = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_preds)
                val_pred_labels = model.predict(X_val)

                # ---- Save in session ----
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['X_columns'] = X.columns

            # ---- Output Results ----
            st.success("✅ Model trained successfully!")
            st.markdown(f"""
            **📊 Model:** XGBoost  
            **🎯 Validation ROC AUC:** `{val_auc:.4f}`  
            **🔢 Training Rows:** `{len(train_df)}`  
            """)

            # ---- Visualizations ----
            st.subheader("📊 Transaction Type Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='type', data=train_df, ax=ax1)
            ax1.set_title("Transaction Types Count")
            st.pyplot(fig1)

            st.subheader("💰 Average Amount per Transaction Type")
            fig2, ax2 = plt.subplots()
            sns.barplot(x='type', y='amount', data=train_df, ax=ax2)
            ax2.set_title("Average Amount per Transaction Type")
            st.pyplot(fig2)

            # ---- Model Performance ----
            st.subheader("📈 Model Performance Metrics (Validation)")

            train_preds = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, train_preds)
            pr_auc = average_precision_score(y_val, val_preds)

            st.markdown(f"""
            **==== XGBoost ====**  
            - **Train ROC AUC:** `{train_auc:.6f}`  
            - **Validation ROC AUC:** `{val_auc:.6f}`  
            - **PR AUC (Precision-Recall):** `{pr_auc:.4f}`  
            """)

            # ---- Classification Report as Table ----
            report_dict = classification_report(y_val, val_pred_labels, digits=4, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.subheader("📑 Classification Report")
            st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

            # ---- Confusion Matrix ----
            st.subheader("🔲 Confusion Matrix on Validation Set")
            fig3, ax3 = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_val, val_pred_labels, ax=ax3, cmap='Blues')
            st.pyplot(fig3)

            # ---- Feature Importance ----
            st.subheader("⭐ Feature Importance")
            fig4, ax4 = plt.subplots(figsize=(8,5))
            sns.barplot(x=model.feature_importances_, y=st.session_state['X_columns'], ax=ax4)
            ax4.set_title("Feature Importance from XGBoost")
            st.pyplot(fig4)

# --- TESTING SECTION ---
if 'model' in st.session_state:
    st.header("2️⃣ Upload Test Dataset for Prediction")

    test_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"], key="test")

    if test_file is not None:
        test_df = pd.read_csv(test_file)
        st.success(f"✅ Test data uploaded. Total rows: {len(test_df)}")

        st.subheader("📄 Full Test Dataset (Before Prediction)")
        st.dataframe(test_df, use_container_width=True)

        if st.button("🔍 Predict Fraud"):
            try:
                # ---- Encode ----
                type_dummies_test = pd.get_dummies(test_df['type'], drop_first=True)
                expected_cols = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                for col in expected_cols:
                    if col not in type_dummies_test:
                        type_dummies_test[col] = 0
                type_dummies_test = type_dummies_test[expected_cols]

                test_df = pd.concat([test_df, type_dummies_test], axis=1)

                # ---- Drop unused ----
                X_test = test_df.drop(['type', 'nameOrig', 'nameDest', 'isFraud'], axis=1, errors='ignore')

                # ---- Scale ----
                X_test_scaled = st.session_state['scaler'].transform(X_test)

                # ---- Predict ----
                preds = st.session_state['model'].predict(X_test_scaled)
                probs = st.session_state['model'].predict_proba(X_test_scaled)[:, 1]

                # ---- Add to DataFrame ----
                test_df['isFraud_Predicted'] = preds
                test_df['Fraud_Probability'] = np.round(probs, 4)

                st.success("✅ Predictions completed!")
                st.subheader("📄 Full Test Dataset (After Prediction)")
                st.dataframe(test_df, use_container_width=True)

                # ---- Download ----
                csv = test_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
