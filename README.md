# 💳 Credit Card Fraud Detection Project

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions
using Machine Learning. Since fraud cases are rare, the dataset is
highly imbalanced and requires special preprocessing techniques.

------------------------------------------------------------------------

## 📊 Dataset Information

-   **Total Records:** 1000
-   **Total Features:** 13
-   **Fraud Transactions:** 14
-   **Genuine Transactions:** 986
-   **Fraud Percentage:** 1.4%

Target Variable: - `0` → Genuine Transaction - `1` → Fraud Transaction

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Imbalanced-learn (SMOTE)
-   Matplotlib

------------------------------------------------------------------------

## 🔄 Project Workflow

1.  Load Dataset
2.  Check Class Imbalance
3.  Feature Scaling using StandardScaler
4.  Handle Imbalance using SMOTE
5.  Train-Test Split (80-20)
6.  Train Random Forest Classifier
7.  Model Evaluation using:
    -   Confusion Matrix
    -   Precision
    -   Recall
    -   F1-Score
    -   ROC-AUC Score

------------------------------------------------------------------------

## 🧠 Why SMOTE?

The dataset is highly imbalanced.\
SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
fraud samples to balance the dataset.

------------------------------------------------------------------------

## 🌳 Model Used

### Random Forest Classifier

-   Ensemble-based algorithm
-   Uses multiple decision trees
-   Final prediction based on majority voting
-   Robust and high performance on classification problems

------------------------------------------------------------------------

## 📈 Evaluation Metrics

Accuracy is not reliable for imbalanced datasets.

Important Metrics: - **Precision** → How many predicted frauds were
correct - **Recall** → How many actual frauds were detected -
**F1-Score** → Balance between Precision & Recall - **ROC-AUC Score** →
Overall model performance

------------------------------------------------------------------------

## ▶️ How to Run the Project

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

------------------------------------------------------------------------

## 🎯 Conclusion

This project demonstrates how to handle imbalanced datasets effectively
using SMOTE and evaluate fraud detection models using appropriate
metrics such as Recall and ROC-AUC instead of relying solely on
accuracy.

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Hyperparameter tuning
-   XGBoost implementation
-   Deep Learning model
-   Deployment using Streamlit
