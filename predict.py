import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from helpers.feature_url_extractor import (
    extract_url_features
)


# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.linear = nn.Linear(
            input_size,
            1
        )

    def forward(self, x):

        return self.linear(x)


# =========================================================
# LOAD LOGISTIC REGRESSION
# =========================================================

LOGISTIC_FEATURES = joblib.load(
    "./notebooks/models/logisticregression/"
    "model_regression_features.pkl"
)

LOGISTIC_SCALER = joblib.load(
    "./notebooks/models/logisticregression/"
    "model_regression_scaler.pkl"
)

logistic_model = LogisticRegressionModel(
    input_size=len(LOGISTIC_FEATURES)
)

logistic_model.load_state_dict(
    torch.load(
        "./notebooks/models/logisticregression/"
        "model_regression.pth",
        map_location=torch.device("cpu")
    )
)

logistic_model.eval()


# =========================================================
# LOAD DECISION TREE
# =========================================================

DECISION_TREE_FEATURES = joblib.load(
    "./notebooks/models/decisiontree/"
    "decision_tree_features.pkl"
)

decision_tree_model = joblib.load(
    "./notebooks/models/decisiontree/"
    "decision_tree_model_b.pkl"
)


#Load KNN Model

knn_model = joblib.load(
    "./notebooks/models/knn/"
    "knn_model.pkl"
)

knn_scaler = joblib.load(
    "./notebooks/models/knn/"
    "knn_scaler.pkl"
)

KNN_FEATURES = joblib.load(
    "./notebooks/models/knn/"
    "knn_features.pkl"
)

# Load Bayes Model

nb_model = joblib.load(
    "./notebooks/models/bayes/"
    "bayes_model.pkl"
)
nb_scaler = joblib.load(
    "./notebooks/models/bayes/"
    "bayes_scaler.pkl"
)
NB_FEATURES = joblib.load(
    "./notebooks/models/bayes/"
    "bayes_features.pkl"
)


#Load Random Forest Features
rf_model = joblib.load(
    "./notebooks/models/randomforest/"
    "rf_model.pkl"
)

RF_FEATURES = joblib.load(
    "./notebooks/models/randomforest/"
    "rf_features.pkl"
)



# =========================================================
# LOGISTIC REGRESSION PREDICTION
# =========================================================

def predict_logistic(url):

    # Extract URL features
    extracted_features = extract_url_features(
        url
    )


    # Arrange in training order
    feature_vector = [
        extracted_features[feature]
        for feature in LOGISTIC_FEATURES
    ]


    # NumPy
    X = np.array(
        feature_vector,
        dtype=np.float32
    ).reshape(1, -1)


    # Scale
    X_scaled = LOGISTIC_SCALER.transform(X)


    # PyTorch tensor
    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )


    # Prediction
    with torch.no_grad():

        logit = logistic_model(
            X_tensor
        )

        probability = torch.sigmoid(
            logit
        ).item()


    # Class
    if probability >= 0.5:

        prediction = "Phishing"

        confidence = (
            probability * 100
        )

    else:

        prediction = "Legitimate"

        confidence = (
            (1 - probability) * 100
        )


    return {
        "prediction": prediction,

        "phishing_probability": round(
            probability * 100,
            2
        ),

        "confidence": round(
            confidence,
            2
        )
    }


# =========================================================
# DECISION TREE PREDICTION
# =========================================================

def predict_decision_tree(url):

    # Extract URL features
    extracted_features = extract_url_features(
        url
    )


    # Arrange in training order
    feature_vector = [
        extracted_features[feature]
        for feature in DECISION_TREE_FEATURES
    ]


    # DataFrame
    X = pd.DataFrame(
        [feature_vector],
        columns=DECISION_TREE_FEATURES
    )


    # Prediction
    prediction_value = (
        decision_tree_model
        .predict(X)[0]
    )


    # Probability
    probabilities = (
        decision_tree_model
        .predict_proba(X)[0]
    )

    phishing_probability = probabilities[1]


    # Class
    if prediction_value == 1:

        prediction = "Phishing"

        confidence = (
            phishing_probability * 100
        )

    else:

        prediction = "Legitimate"

        confidence = (
            (1 - phishing_probability) * 100
        )


    return {
        "prediction": prediction,

        "phishing_probability": round(
            phishing_probability * 100,
            2
        ),

        "confidence": round(
            confidence,
            2
        )
    }


def predict_knn(url):

    # Extract features from URL
    extracted_features = extract_url_features(url)

    # Keep exact feature order used during training
    feature_vector = [
        extracted_features[feature]
        for feature in KNN_FEATURES
    ]

    # Convert to DataFrame
    X = pd.DataFrame(
        [feature_vector],
        columns=KNN_FEATURES
    )

    # Scale
    X_scaled = knn_scaler.transform(X)

    # Prediction
    prediction_value = knn_model.predict(X_scaled)[0]

    # Probability
    probability = knn_model.predict_proba(
        X_scaled
    )[0][1]

    # Convert to human-readable result
    if prediction_value == 1:

        prediction = "Phishing"
        confidence = probability * 100

    else:

        prediction = "Legitimate"
        confidence = (1 - probability) * 100

    return {
        "url": url,
        "prediction": prediction,
        "phishing_probability": round(
            probability * 100,
            2
        ),
        "confidence": round(
            confidence,
            2
        ),
        "features": extracted_features
    }

def predict_naive_bayes(url):

    # ---------------------------------------------
    # Extract URL features
    # ---------------------------------------------

    extracted_features = extract_url_features(url)


    # ---------------------------------------------
    # Maintain exact feature order
    # ---------------------------------------------

    feature_vector = [
        extracted_features[feature]
        for feature in NB_FEATURES
    ]


    # ---------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------

    X = pd.DataFrame(
        [feature_vector],
        columns=NB_FEATURES
    )


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction_value = nb_model.predict(X)[0]


    # ---------------------------------------------
    # Probability
    # ---------------------------------------------

    probability = nb_model.predict_proba(X)[0][1]


    # ---------------------------------------------
    # Human-readable prediction
    # ---------------------------------------------

    if prediction_value == 1:

        prediction = "Phishing"

        confidence = probability * 100

    else:

        prediction = "Legitimate"

        confidence = (1 - probability) * 100


    return {
        "url": url,

        "prediction": prediction,

        "phishing_probability": round(
            probability * 100,
            2
        ),

        "confidence": round(
            confidence,
            2
        ),

        "features": extracted_features
    }


def predict_random_forest(url):

    # ---------------------------------------------
    # Extract URL features
    # ---------------------------------------------

    extracted_features = extract_url_features(url)


    # ---------------------------------------------
    # Maintain exact feature order
    # ---------------------------------------------

    feature_vector = [
        extracted_features[feature]
        for feature in RF_FEATURES
    ]


    # ---------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------

    X = pd.DataFrame(
        [feature_vector],
        columns=RF_FEATURES
    )


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction_value = rf_model.predict(X)[0]


    # ---------------------------------------------
    # Probability
    # ---------------------------------------------

    probability = rf_model.predict_proba(
        X
    )[0][1]


    # ---------------------------------------------
    # Human-readable result
    # ---------------------------------------------

    if prediction_value == 1:

        prediction = "Phishing"

        confidence = probability * 100

    else:

        prediction = "Legitimate"

        confidence = (1 - probability) * 100


    return {
        "url": url,

        "prediction": prediction,

        "phishing_probability": round(
            probability * 100,
            2
        ),

        "confidence": round(
            confidence,
            2
        ),

        "features": extracted_features
    }

# =========================================================
# UNIFIED PREDICTION FUNCTION
# =========================================================

def predict_url(url, model_name):

    if model_name == "logistic_regression":

        result = predict_logistic(url)

        result["model"] = (
            "Logistic Regression"
        )

        return result


    elif model_name == "decision_tree":

        result = predict_decision_tree(url)

        result["model"] = (
            "Decision Tree"
        )

        return result

    elif model_name == "knn":

        result = predict_knn(url)

        result["model"] = "KNN"

        return result

    elif model_name == "naive_bayes":

        result = predict_naive_bayes(url)

        result["model"] = "Naive Bayes"

        return result

    elif model_name == "random_forest":
        print("Inside Random Forest")

        result = predict_random_forest(url)

        result["model"] = "Random Forest"

        return result

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

