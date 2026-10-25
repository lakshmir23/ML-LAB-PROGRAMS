import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("diabetes.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_no_scale = LogisticRegression(max_iter=1000)
model_no_scale.fit(X_train, y_train)
y_pred_no_scale = model_no_scale.predict(X_test)

metrics_no_scale = {
    "Accuracy": accuracy_score(y_test, y_pred_no_scale),
    "Precision": precision_score(y_test, y_pred_no_scale, zero_division=0),
    "Recall": recall_score(y_test, y_pred_no_scale, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_no_scale, zero_division=0)
}

print("Without Scaling:")
print(metrics_no_scale)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

metrics_scaled = {
    "Accuracy": accuracy_score(y_test, y_pred_scaled),
    "Precision": precision_score(y_test, y_pred_scaled, zero_division=0),
    "Recall": recall_score(y_test, y_pred_scaled, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_scaled, zero_division=0)
}

print("\nWith Scaling:")
print(metrics_scaled)
