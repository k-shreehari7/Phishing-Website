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

# ============================================================
# IMPORT YOUR EXISTING PREDICTION MODULE
# ============================================================

from predict import (
    predict_url,
    extract_url_features,

    # Logistic Regression
    logistic_model,
    LOGISTIC_FEATURES,
    LOGISTIC_SCALER,

    # Decision Tree
    decision_tree_model,
    DECISION_TREE_FEATURES,

    # KNN
    knn_model,
    knn_scaler,
    KNN_FEATURES,

    # Naive Bayes
    nb_model,
    NB_FEATURES,

    # Random Forest
    rf_model,
    RF_FEATURES
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🛡️ PhishGuard")

st.write(
    "Phishing Website Detection using Machine Learning"
)

st.write(
    "This application uses Logistic Regression, Decision Tree, "
    "K-Nearest Neighbors, Naive Bayes and Random Forest models "
    "to detect phishing websites."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "🔎 URL Prediction",
        "📊 Test Dataset Evaluation",
        "📈 Model Comparison"
    ]
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest"
}


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_label(value):
    """
    Convert different possible target-label formats
    into binary values.

    0 = Legitimate
    1 = Phishing
    """

    if pd.isna(value):
        return np.nan

    # Numeric labels
    if isinstance(value, (int, float, np.integer, np.floating)):

        if value in [0, 1]:
            return int(value)

    # String labels
    value = str(value).strip().lower()

    label_mapping = {
        "legitimate": 0,
        "phishing": 1,
        "legit": 0,
        "phish": 1,
        "benign": 0,
        "malicious": 1,
        "safe": 0,
        "unsafe": 1,
        "0": 0,
        "1": 1
    }

    return label_mapping.get(value, np.nan)


# ============================================================
# FUNCTION TO FIND TARGET COLUMN
# ============================================================

def find_target_column(df):

    possible_columns = [
        "status",
        "label",
        "target",
        "class",
        "y"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


# ============================================================
# FUNCTION TO FIND URL COLUMN
# ============================================================

def find_url_column(df):

    possible_columns = [
        "url",
        "URL",
        "Url"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


# ============================================================
# FUNCTION TO GET MODEL PREDICTION
# ============================================================

def predict_single_url(url, model_name):

    features = extract_url_features(url)

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    if model_name == "logistic_regression":

        feature_vector = [
            features[feature]
            for feature in LOGISTIC_FEATURES
        ]

        X = pd.DataFrame(
            [feature_vector],
            columns=LOGISTIC_FEATURES
        )

        X_scaled = LOGISTIC_SCALER.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():

            logit = logistic_model(X_tensor)

            probability = torch.sigmoid(
                logit
            ).item()

        prediction = (
            1 if probability >= 0.5 else 0
        )

    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    elif model_name == "decision_tree":

        feature_vector = [
            features[feature]
            for feature in DECISION_TREE_FEATURES
        ]

        X = pd.DataFrame(
            [feature_vector],
            columns=DECISION_TREE_FEATURES
        )

        prediction = int(
            decision_tree_model.predict(X)[0]
        )

        probability = (
            decision_tree_model
            .predict_proba(X)[0][1]
        )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    elif model_name == "knn":

        feature_vector = [
            features[feature]
            for feature in KNN_FEATURES
        ]

        X = pd.DataFrame(
            [feature_vector],
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

    # --------------------------------------------------------
    # NAIVE BAYES
    # --------------------------------------------------------

    elif model_name == "naive_bayes":

        feature_vector = [
            features[feature]
            for feature in NB_FEATURES
        ]

        X = pd.DataFrame(
            [feature_vector],
            columns=NB_FEATURES
        )

        prediction = int(
            nb_model.predict(X)[0]
        )

        probability = (
            nb_model
            .predict_proba(X)[0][1]
        )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    elif model_name == "random_forest":

        feature_vector = [
            features[feature]
            for feature in RF_FEATURES
        ]

        X = pd.DataFrame(
            [feature_vector],
            columns=RF_FEATURES
        )

        prediction = int(
            rf_model.predict(X)[0]
        )

        probability = (
            rf_model
            .predict_proba(X)[0][1]
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return prediction, probability


# ============================================================
# FUNCTION TO EVALUATE A MODEL ON CSV
# ============================================================

def evaluate_model(df, model_name, url_column, target_column):

    predictions = []
    probabilities = []

    errors = []

    total = len(df)

    # --------------------------------------------------------
    # PROCESS EVERY URL
    # --------------------------------------------------------

    for index, url in enumerate(df[url_column]):

        try:

            prediction, probability = predict_single_url(
                str(url),
                model_name
            )

            predictions.append(prediction)
            probabilities.append(probability)

        except Exception as e:

            errors.append(
                f"Row {index}: {url} -> {str(e)}"
            )

            predictions.append(np.nan)
            probabilities.append(np.nan)

    # --------------------------------------------------------
    # CREATE RESULT DATAFRAME
    # --------------------------------------------------------

    results = df.copy()

    results["prediction"] = predictions
    results["phishing_probability"] = probabilities

    # --------------------------------------------------------
    # CONVERT ACTUAL LABELS
    # --------------------------------------------------------

    results["actual_label"] = (
        results[target_column]
        .apply(convert_label)
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    valid = (
        results["actual_label"].notna()
        &
        results["prediction"].notna()
        &
        results["phishing_probability"].notna()
    )

    valid_results = results[valid].copy()

    # --------------------------------------------------------
    # CHECK FOR EMPTY DATA
    # --------------------------------------------------------

    if len(valid_results) == 0:

        return {
            "success": False,
            "error": (
                "No valid samples were available for evaluation. "
                "Please check the target column and labels."
            ),
            "results": results,
            "errors": errors
        }

    # --------------------------------------------------------
    # TRUE / PREDICTED VALUES
    # --------------------------------------------------------

    y_true = (
        valid_results["actual_label"]
        .astype(int)
        .values
    )

    y_pred = (
        valid_results["prediction"]
        .astype(int)
        .values
    )

    y_prob = (
        valid_results["phishing_probability"]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # CHECK BOTH CLASSES
    # --------------------------------------------------------

    unique_classes = np.unique(y_true)

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    if len(unique_classes) == 2:

        auc = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        auc = np.nan

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
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

    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {
        "success": True,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "classification_report": report_df,
        "results": results,
        "valid_results": valid_results,
        "errors": errors
    }


# ============================================================
# PAGE 1
# URL PREDICTION
# ============================================================

if page == "🔎 URL Prediction":

    st.header("🔎 Phishing URL Detection")

    st.write(
        "Enter a website URL and select a machine learning "
        "model to predict whether the website is legitimate "
        "or phishing."
    )

    st.divider()

    # --------------------------------------------------------
    # URL INPUT
    # --------------------------------------------------------

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com"
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model_display = st.selectbox(
        "Select Machine Learning Model",
        list(MODEL_OPTIONS.keys())
    )

    selected_model = MODEL_OPTIONS[
        selected_model_display
    ]

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Check URL",
        use_container_width=True
    ):

        if not url.strip():

            st.warning(
                "Please enter a website URL."
            )

        else:

            try:

                prediction, probability = (
                    predict_single_url(
                        url,
                        selected_model
                    )
                )

                st.divider()

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                if prediction == 1:

                    st.error(
                        "🚨 PHISHING WEBSITE"
                    )

                    prediction_text = "Phishing"

                    confidence = probability * 100

                else:

                    st.success(
                        "✅ LEGITIMATE WEBSITE"
                    )

                    prediction_text = "Legitimate"

                    confidence = (
                        1 - probability
                    ) * 100

                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Selected Model",
                        selected_model_display
                    )

                with col2:

                    st.metric(
                        "Prediction",
                        prediction_text
                    )

                with col3:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                st.divider()

                # ------------------------------------------------
                # PROBABILITY
                # ------------------------------------------------

                st.subheader(
                    "Phishing Probability"
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                st.write(
                    f"{probability * 100:.2f}%"
                )

            except Exception as e:

                st.error(
                    "An error occurred while predicting the URL."
                )

                st.exception(e)


# ============================================================
# PAGE 2
# TEST DATASET EVALUATION
# ============================================================

elif page == "📊 Test Dataset Evaluation":

    st.header(
        "📊 Test Dataset Evaluation"
    )

    st.write(
        "Upload your test CSV to evaluate one of the "
        "trained machine learning models."
    )

    st.info(
        "The CSV must contain a URL column and a target "
        "column such as 'status'."
    )

    st.divider()

    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Please upload a CSV file to continue."
        )

    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                "Could not read the uploaded CSV."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        st.success(
            "Dataset uploaded successfully."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1]
            )

        # ----------------------------------------------------
        # DATASET PREVIEW
        # ----------------------------------------------------

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DISPLAY COLUMNS
        # ----------------------------------------------------

        with st.expander(
            "View Dataset Columns"
        ):

            st.write(
                df.columns.tolist()
            )

        # ----------------------------------------------------
        # FIND URL COLUMN
        # ----------------------------------------------------

        url_column = find_url_column(df)

        if url_column is None:

            st.error(
                """
                No URL column was found.

                Your CSV must contain a column named:
                url
                """
            )

            st.stop()

        # ----------------------------------------------------
        # FIND TARGET COLUMN
        # ----------------------------------------------------

        target_column = find_target_column(df)

        if target_column is None:

            st.error(
                """
                No target column was found.

                Your CSV must contain a target column such as:
                status
                label
                target
                class
                """
            )

            st.stop()

        # ----------------------------------------------------
        # SHOW TARGET VALUES
        # ----------------------------------------------------

        st.write(
            f"**URL column:** `{url_column}`"
        )

        st.write(
            f"**Target column:** `{target_column}`"
        )

        st.write(
            "**Target values found:**"
        )

        st.write(
            df[target_column]
            .value_counts(dropna=False)
        )

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------

        selected_model_display = st.selectbox(
            "Select Model",
            list(MODEL_OPTIONS.keys())
        )

        selected_model = MODEL_OPTIONS[
            selected_model_display
        ]

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        if st.button(
            "📈 Evaluate Model",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting URL features and evaluating model..."
            ):

                evaluation = evaluate_model(
                    df,
                    selected_model,
                    url_column,
                    target_column
                )

            # ------------------------------------------------
            # CHECK RESULT
            # ------------------------------------------------

            if not evaluation["success"]:

                st.error(
                    evaluation["error"]
                )

                st.write(
                    "Actual values found:"
                )

                st.write(
                    df[target_column]
                    .unique()
                    .tolist()
                )

                st.stop()

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                f"{selected_model_display} evaluation completed."
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            st.subheader(
                "Evaluation Metrics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Accuracy",
                    f"{evaluation['accuracy']:.4f}"
                )

                st.metric(
                    "Precision",
                    f"{evaluation['precision']:.4f}"
                )

            with col2:

                st.metric(
                    "Recall",
                    f"{evaluation['recall']:.4f}"
                )

                st.metric(
                    "F1 Score",
                    f"{evaluation['f1']:.4f}"
                )

            with col3:

                if np.isnan(
                    evaluation["auc"]
                ):

                    auc_text = "N/A"

                else:

                    auc_text = (
                        f"{evaluation['auc']:.4f}"
                    )

                st.metric(
                    "ROC-AUC",
                    auc_text
                )

                st.metric(
                    "MCC",
                    f"{evaluation['mcc']:.4f}"
                )

            st.divider()

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            st.subheader(
                "Confusion Matrix"
            )

            cm = evaluation[
                "confusion_matrix"
            ]

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual Legitimate",
                    "Actual Phishing"
                ],
                columns=[
                    "Predicted Legitimate",
                    "Predicted Phishing"
                ]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            # ------------------------------------------------
            # CLASSIFICATION REPORT
            # ------------------------------------------------

            st.subheader(
                "Classification Report"
            )

            st.dataframe(
                evaluation[
                    "classification_report"
                ],
                use_container_width=True
            )

            # ------------------------------------------------
            # PREDICTION DATA
            # ------------------------------------------------

            st.subheader(
                "Prediction Results"
            )

            display_columns = [
                url_column,
                target_column,
                "actual_label",
                "prediction",
                "phishing_probability"
            ]

            available_columns = [
                column
                for column in display_columns
                if column in evaluation["results"].columns
            ]

            result_display = (
                evaluation["results"]
                [available_columns]
                .copy()
            )

            result_display[
                "prediction"
            ] = result_display[
                "prediction"
            ].map({
                0: "Legitimate",
                1: "Phishing"
            })

            result_display[
                "actual_label"
            ] = result_display[
                "actual_label"
            ].map({
                0: "Legitimate",
                1: "Phishing"
            })

            result_display[
                "phishing_probability"
            ] = (
                result_display[
                    "phishing_probability"
                ] * 100
            ).round(2)

            st.dataframe(
                result_display,
                use_container_width=True
            )

            # ------------------------------------------------
            # PROCESSING ERRORS
            # ------------------------------------------------

            if evaluation["errors"]:

                st.warning(
                    f"{len(evaluation['errors'])} "
                    "URLs could not be processed."
                )

                with st.expander(
                    "View processing errors"
                ):

                    for error in evaluation[
                        "errors"
                    ]:

                        st.write(error)


# ============================================================
# PAGE 3
# MODEL COMPARISON
# ============================================================

elif page == "📈 Model Comparison":

    st.header(
        "📈 Model Performance Comparison"
    )

    st.write(
        "Compare all five trained models using the "
        "same uploaded test dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        key="comparison_upload"
    )

    if uploaded_file is None:

        st.info(
            "Upload your test CSV to compare the models."
        )

    else:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                "Could not read the uploaded CSV."
            )

            st.exception(e)

            st.stop()

        # ----------------------------------------------------
        # FIND COLUMNS
        # ----------------------------------------------------

        url_column = find_url_column(df)

        target_column = find_target_column(df)

        if url_column is None:

            st.error(
                "No 'url' column found in the CSV."
            )

            st.stop()

        if target_column is None:

            st.error(
                "No target column found in the CSV."
            )

            st.stop()

        # ----------------------------------------------------
        # TARGET INFORMATION
        # ----------------------------------------------------

        st.write(
            f"URL column: `{url_column}`"
        )

        st.write(
            f"Target column: `{target_column}`"
        )

        st.write(
            "Target values:"
        )

        st.write(
            df[target_column]
            .value_counts(dropna=False)
        )

        # ----------------------------------------------------
        # RUN COMPARISON
        # ----------------------------------------------------

        if st.button(
            "🚀 Compare All Models",
            use_container_width=True
        ):

            comparison_results = []

            progress = st.progress(0)

            models = list(
                MODEL_OPTIONS.items()
            )

            for index, (
                display_name,
                model_name
            ) in enumerate(models):

                with st.spinner(
                    f"Evaluating {display_name}..."
                ):

                    evaluation = evaluate_model(
                        df,
                        model_name,
                        url_column,
                        target_column
                    )

                if evaluation["success"]:

                    comparison_results.append({

                        "ML Model": display_name,

                        "Accuracy":
                            evaluation["accuracy"],

                        "AUC":
                            evaluation["auc"],

                        "Precision":
                            evaluation["precision"],

                        "Recall":
                            evaluation["recall"],

                        "F1 Score":
                            evaluation["f1"],

                        "MCC":
                            evaluation["mcc"]
                    })

                else:

                    comparison_results.append({

                        "ML Model": display_name,

                        "Accuracy": np.nan,

                        "AUC": np.nan,

                        "Precision": np.nan,

                        "Recall": np.nan,

                        "F1 Score": np.nan,

                        "MCC": np.nan
                    })

                progress.progress(
                    (index + 1) / len(models)
                )

            # ------------------------------------------------
            # CREATE COMPARISON TABLE
            # ------------------------------------------------

            comparison_df = pd.DataFrame(
                comparison_results
            )

            st.success(
                "All models evaluated successfully."
            )

            st.subheader(
                "Model Comparison"
            )

            st.dataframe(
                comparison_df.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "AUC": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1 Score": "{:.4f}",
                        "MCC": "{:.4f}"
                    }
                ),
                use_container_width=True
            )

            # ------------------------------------------------
            # BEST MODEL
            # ------------------------------------------------

            valid_comparison = (
                comparison_df
                .dropna(subset=["F1 Score"])
            )

            if len(valid_comparison) > 0:

                best_model = (
                    valid_comparison
                    .sort_values(
                        "F1 Score",
                        ascending=False
                    )
                    .iloc[0]
                )

                st.divider()

                st.subheader(
                    "🏆 Overall Best Model"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Best Model",
                        best_model["ML Model"]
                    )

                with col2:

                    st.metric(
                        "F1 Score",
                        f"{best_model['F1 Score']:.4f}"
                    )

                with col3:

                    st.metric(
                        "Accuracy",
                        f"{best_model['Accuracy']:.4f}"
                    )

                st.info(
                    "The overall winner is selected based on "
                    "the highest F1 Score."
                )