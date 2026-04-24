# ==========================================
# Iris Dataset Classification Pipeline
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================
# Load Dataset
# ==========================================

def load_data():
    data = pd.read_csv("datasets/Iris_data_sample.csv")
    print("Dataset Loaded:", data.shape)
    return data


# ==========================================
# Data Preprocessing
# ==========================================

def clean_data(data):
    # Convert columns to numeric
    data['SepalLengthCm'] = pd.to_numeric(data['SepalLengthCm'], errors='coerce')
    data['PetalLengthCm'] = pd.to_numeric(data['PetalLengthCm'], errors='coerce')

    # Remove irrelevant column
    data = data.drop(columns=['Unnamed: 0'])

    # Handle missing values
    data['SepalLengthCm'] = data['SepalLengthCm'].fillna(data['SepalLengthCm'].mean())
    data['SepalWidthCm'] = data['SepalWidthCm'].fillna(data['SepalWidthCm'].mean())
    data['PetalLengthCm'] = data['PetalLengthCm'].fillna(data['PetalLengthCm'].median())
    data['Species'] = data['Species'].fillna(data['Species'].mode()[0])

    print("Preprocessing Completed")
    return data


# ==========================================
# Train Models
# ==========================================

def train_models(data):
    X = data.drop("Species", axis=1)
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    return y_test, y_pred_log, y_pred_knn


# ==========================================
# Evaluation
# ==========================================

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n{model_name} Performance")
    print("-" * 40)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ==========================================
# Main Execution
# ==========================================

def main():
    data = load_data()
    data = clean_data(data)

    y_test, y_pred_log, y_pred_knn = train_models(data)

    evaluate_model(y_test, y_pred_log, "Logistic Regression")
    evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")


if __name__ == "__main__":
    main()