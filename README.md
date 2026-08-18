# Phishing Website Classification

## a. Problem Statement

Phishing websites are designed to deceive users into revealing sensitive information such as
login credentials, financial details, and personal data. Detecting phishing websites automatically
can help improve web security and reduce the risk of users accessing malicious websites.

The objective of this project is to build and compare multiple machine learning classification
models for identifying whether a given URL belongs to a **legitimate** or **phishing** website.

The following classification models are implemented and evaluated using multiple performance
metrics:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (Ensemble)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews
Correlation Coefficient (MCC).

---

## b. Dataset Description

The dataset used in this project is a **Phishing Website Dataset** containing URL-based
features extracted from websites.

- **Number of instances:** 11,430
- **Number of columns:** 89
- **Features:** 87 numerical URL-related features
- **URL column:** `url`
- **Target column:** `status`
- **Classes:** `legitimate` and `phishing`
- **Class distribution:**
  - Legitimate: 5,715
  - Phishing: 5,715

The dataset is balanced, with an equal number of legitimate and phishing website samples.

The features describe different characteristics of URLs, such as:

- URL length
- Number of dots
- Number of slashes
- Number of query marks
- Number of digits
- Number of subdomains
- Presence of an IP address
- Presence of phishing-related keywords
- Prefix/suffix characteristics
- Hostname characteristics
- Path characteristics

The target variable is converted into a binary classification:

- `legitimate` → `0`
- `phishing` → `1`

For model evaluation, the dataset is divided into training and testing sets using an
80:20 train-test split.

---

## c. GitHub Repository Link

The complete source code, trained models, helper functions, test data, and project files are
available in the following GitHub repository:

**GitHub Repository:**  
https://github.com/k-shreehari7/Phishing-Website

The repository contains:

- Source code
- Model training notebooks
- Saved trained models
- Feature extraction functions
- Test dataset
- Streamlit application
- `requirements.txt`
- `README.md`

---

## d. Models Used

Five classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)

The following metrics were used for comparison:

- **Accuracy:** Overall percentage of correctly classified samples.
- **AUC:** Measures the model's ability to distinguish between legitimate and phishing websites.
- **Precision:** Proportion of predicted phishing websites that were actually phishing.
- **Recall:** Proportion of actual phishing websites correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Balanced classification metric that considers all four confusion-matrix categories.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | XX.XX% | XX.XX | XX.XX% | XX.XX% | XX.XX% | XX.XX |
| Decision Tree | 85.04% | XX.XX | XX.XX% | 86.70% | 85.28% | XX.XX |
| kNN | 88.01% | XX.XX | XX.XX% | 86.79% | 87.87% | XX.XX |
| Naive Bayes | 70.82% | XX.XX | 92.81% | 45.14% | 60.74% | XX.XX |
| Random Forest (Ensemble) | 89.68% | XX.XX | XX.XX% | 89.33% | 89.64% | XX.XX |

> **Note:** Replace the `XX.XX` values with the AUC, Precision, and MCC values obtained
> from your final model evaluation. These values should be calculated on the same test set
> used for the other metrics.

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression provides a simple linear baseline for the phishing classification problem. Its performance is useful as a reference for comparing more complex models. |
| **Decision Tree** | The Decision Tree performs better than the basic linear approach by learning non-linear relationships between URL features. Increasing the tree depth improves performance initially, but deeper trees can increase the risk of overfitting. |
| **kNN** | kNN achieves strong classification performance after feature scaling. With the selected value of `k`, it provides good accuracy and F1 score, making it a competitive model for this dataset. |
| **Naive Bayes** | Naive Bayes achieves high precision but significantly lower recall. This means that when it predicts a website as phishing, it is usually correct, but it misses a considerable number of actual phishing websites. |
| **Random Forest (Ensemble)** | Random Forest achieves the best overall performance among the evaluated models. Its ensemble of decision trees allows it to capture complex non-linear relationships between URL features while providing strong accuracy, recall, and F1 score. |

### Overall Winner for Your Dataset

**Random Forest (Ensemble)** is the overall winner for this dataset.

It achieved the highest observed performance with:

- **Accuracy:** 89.68%
- **Recall:** 89.33%
- **F1 Score:** 89.64%

Random Forest provides a strong balance between correctly identifying phishing websites
and avoiding false classifications. Its high recall is particularly important for phishing
detection because failing to identify an actual phishing website can pose a security risk.

---

## Conclusion

The experimental results show that ensemble-based learning performs particularly well for
the selected phishing website dataset.

Among the evaluated models, Random Forest provided the best overall balance of accuracy,
precision, recall, and F1 score. kNN also demonstrated strong performance, while Decision
Tree provided a reasonable balance between model complexity and classification performance.

Naive Bayes achieved high precision but had substantially lower recall, making it less suitable
when detecting as many phishing websites as possible is the primary objective.

Therefore, **Random Forest is selected as the final model for the phishing website
classification application.**