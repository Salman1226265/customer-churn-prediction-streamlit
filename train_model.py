import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------------- Load dataset ----------------
df = pd.read_csv("Data/Customers Dataset.csv")

print(df.columns)

# ---------------- Check churn distribution ----------------
counts = df['Churn'].value_counts()
print(counts)

churn_percentage = (df['Churn'].value_counts(normalize=True) * 100).round(2)
print("Percentage:")
print(churn_percentage)

# ---------------- Encode categorical features ----------------
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

subsType_encoder = LabelEncoder()
df["SubscriptionType"] = subsType_encoder.fit_transform(df["SubscriptionType"])

contract_encoder = LabelEncoder()
df["ContractLength"] = contract_encoder.fit_transform(df["ContractLength"])

# ---------------- Features & Target ----------------
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# ---------------- Scale numerical features ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Train Model ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# ---------------- Save EVERYTHING ----------------
with open("model.pkl", "wb") as file:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "gender_encoder": gender_encoder,
        "subsType_encoder": subsType_encoder,
        "contract_encoder": contract_encoder,
        "feature_columns": X.columns.tolist()
    }, file)

print("model.pkl file created successfully with encoders & scaler")

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances)