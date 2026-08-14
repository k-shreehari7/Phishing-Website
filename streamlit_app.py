import streamlit as st
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from predict import (
    predict_url,

    logistic_model,
    LOGISTIC_FEATURES,
    LOGISTIC_SCALER,

    decision_tree_model,
    DECISION_TREE_FEATURES,

    knn_model,
    knn_scaler,
    KNN_FEATURES,

    nb_model,
    NB_FEATURES,

    rf_model,
    RF_FEATURES,

    extract_url_features
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ PhishGuard")
st.subheader("Phishing Website Detection using Machine Learning")

st.write(
    "Enter a URL to detect whether it is potentially phishing "
    "or legitimate using multiple machine learning models."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select an option",
    [
        "🔎 URL Prediction",
        "📊 Test Dataset Evaluation"
    ]
)


# =========================================================
# URL PREDICTION
# =========================================================

if page == "🔎 URL Prediction":

    st.header("🔎 Check a Website URL")

    url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com"
    )

    model_options = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest"
    }

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        list(model_options.keys())
    )

    if st.button("🔍 Check URL"):

        if not url.strip():

            st.warning("Please enter a URL.")

        else:

            model_name = model_options[selected_model]

            try:

                result = predict_url(
                    url,
                    model_name
                )

                st.divider()

                st.subheader("Prediction Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Model",
                        result["model"]
                    )

                with col2:
                    st.metric(
                        "Prediction",
                        result["prediction"]
                    )

                with col3:
                    st.metric(
                        "Confidence",
                        f'{result["confidence"]:.2f}%'
                    )

                st.divider()

                if result["prediction"] == "Phishing":

                    st.error(
                        f"🚨 This URL is predicted to be **PHISHING**"
                    )

                else:

                    st.success(
                        f"✅ This URL is predicted to be **LEGITIMATE**"
                    )

                st.metric(
                    "Phishing Probability",
                    f'{result["phishing_probability"]:.2f}%'
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {str(e)}"
                )


# =========================================================
# DATASET EVALUATION
# =========================================================

elif page == "📊 Test Dataset Evaluation":

    st.header("📊 Test Dataset Evaluation")

    st.write(
        "Upload a CSV containing the test URLs and their actual labels."
    )

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded successfully: {df.shape[0]} rows"
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.write(
            f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        )

        # -------------------------------------------------
        # FIND URL COLUMN
        # -------------------------------------------------

        possible_url_columns = [
            "url",
            "URL",
            "Url"
        ]

        url_column = None

        for column in possible_url_columns:

            if column in df.columns:
                url_column = column
                break

        if url_column is None:

            st.error(
                "CSV must contain a 'url' column."
            )

            st.stop()

        # -------------------------------------------------
        # FIND TARGET COLUMN
        # -------------------------------------------------

        possible_target_columns = [
            "status",
            "label",
            "target",
            "class"
        ]

        target_column = None

        for column in possible_target_columns:

            if column in df.columns:
                target_column = column
                break

        if target_column is None:

            st.error(
                "CSV must contain a target column such as "
                "'status', 'label', or 'target'."
            )

            st.stop()

        st.info(
            f"URL column: `{url_column}` | "
            f"Target column: `{target_column}`"
        )

        # -------------------------------------------------
        # MODEL SELECTION
        # -------------------------------------------------

        model_options = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "knn",
            "Naive Bayes": "naive_bayes",
            "Random Forest": "random_forest"
        }

        selected_model = st.selectbox(
            "Select Model for Evaluation",
            list(model_options.keys())
        )

        if st.button("📈 Evaluate Model"):

            model_name = model_options[selected_model]

            progress = st.progress(0)

            predictions = []
            probabilities = []

            total = len(df)

            # ---------------------------------------------
            # EXTRACT FEATURES
            # ---------------------------------------------

            for i, url in enumerate(df[url_column]):

                try:

                    features = extract_url_features(
                        str(url)
                    )

                    # =====================================
                    # LOGISTIC REGRESSION
                    # =====================================

                    if model_name == "logistic_regression":

                        vector = [
                            features[f]
                            for f in LOGISTIC_FEATURES
                        ]

                        X = np.array(
                            vector,
                            dtype=np.float32
                        ).reshape(1, -1)

                        X_scaled = LOGISTIC_SCALER.transform(X)

                        X_tensor = torch.tensor(
                            X_scaled,
                            dtype=torch.float32
                        )

                        with torch.no_grad():

                            logit = logistic_model(
                                X_tensor
                            )

                            probability = torch.sigmoid(
                                logit
                            ).item()

                        prediction = (
                            1 if probability >= 0.5 else 0
                        )

                    # =====================================
                    # DECISION TREE
                    # =====================================

                    elif model_name == "decision_tree":

                        vector = [
                            features[f]
                            for f in DECISION_TREE_FEATURES
                        ]

                        X = pd.DataFrame(
                            [vector],
                            columns=DECISION_TREE_FEATURES
                        )

                        prediction = int(
                            decision_tree_model.predict(X)[0]
                        )

                        probability = (
                            decision_tree_model
                            .predict_proba(X)[0][1]
                        )

                    # =====================================
                    # KNN
                    # =====================================

                    elif model_name == "knn":

                        vector = [
                            features[f]
                            for f in KNN_FEATURES
                        ]

                        X = pd.DataFrame(
                            [vector],
                            columns=KNN_FEATURES
                        )

                        X_scaled = knn_scaler.transform(X)

                        prediction = int(
                            knn_model.predict(X_scaled)[0]
                        )

                        probability = (
                            knn_model
                            .predict_proba(X_scaled)[0][1]
                        )

                    # =====================================
                    # NAIVE BAYES
                    # =====================================

                    elif model_name == "naive_bayes":

                        vector = [
                            features[f]
                            for f in NB_FEATURES
                        ]

                        X = pd.DataFrame(
                            [vector],
                            columns=NB_FEATURES
                        )

                        prediction = int(
                            nb_model.predict(X)[0]
                        )

                        probability = (
                            nb_model
                            .predict_proba(X)[0][1]
                        )

                    # =====================================
                    # RANDOM FOREST
                    # =====================================

                    elif model_name == "random_forest":

                        vector = [
                            features[f]
                            for f in RF_FEATURES
                        ]

                        X = pd.DataFrame(
                            [vector],
                            columns=RF_FEATURES
                        )

                        prediction = int(
                            rf_model.predict(X)[0]
                        )

                        probability = (
                            rf_model
                            .predict_proba(X)[0][1]
                        )

                    predictions.append(prediction)
                    probabilities.append(probability)

                except Exception as e:

                    st.warning(
                        f"Could not process URL: {url}"
                    )

                    predictions.append(0)
                    probabilities.append(0.0)

                progress.progress(
                    (i + 1) / total
                )

            # -------------------------------------------------
            # CONVERT ACTUAL LABELS
            # -------------------------------------------------

            y_true_raw = df[target_column]

            if y_true_raw.dtype == "object":

                y_true = (
                    y_true_raw
                    .astype(str)
                    .str.lower()
                    .map({
                        "legitimate": 0,
                        "phishing": 1
                    })
                )

            else:

                y_true = pd.to_numeric(
                    y_true_raw,
                    errors="coerce"
                )

            # Remove invalid labels

            valid = y_true.notna()

            y_true = y_true[valid].astype(int)

            y_pred = np.array(predictions)[valid.values]

            y_prob = np.array(probabilities)[valid.values]

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            accuracy = accuracy_score(
                y_true,
                y_pred
            )

            precision = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            auc = roc_auc_score(
                y_true,
                y_prob
            )

            mcc = matthews_corrcoef(
                y_true,
                y_pred
            )

            # -------------------------------------------------
            # DISPLAY RESULTS
            # -------------------------------------------------

            st.success(
                f"{selected_model} evaluation completed!"
            )

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )

                st.metric(
                    "Precision",
                    f"{precision:.4f}"
                )

            with col2:

                st.metric(
                    "Recall",
                    f"{recall:.4f}"
                )

                st.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )

            with col3:

                st.metric(
                    "ROC-AUC",
                    f"{auc:.4f}"
                )

                st.metric(
                    "MCC",
                    f"{mcc:.4f}"
                )

            # -------------------------------------------------
            # CONFUSION MATRIX
            # -------------------------------------------------

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Legitimate", "Actual Phishing"],
                columns=["Predicted Legitimate", "Predicted Phishing"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            # -------------------------------------------------
            # CLASSIFICATION REPORT
            # -------------------------------------------------

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                target_names=[
                    "Legitimate",
                    "Phishing"
                ],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df,
                use_container_width=True
            )