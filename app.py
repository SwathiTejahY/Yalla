
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Show Python version
st.write("🚨 Python version in use:", sys.version)

# Title
st.title("Multi-Classifier App (Best model labeled as SLSTM)")

# Load dataset
@st.cache_data
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")
    target_names = data.target_names
    return X, y, target_names

X, y, target_names = load_data()
st.write("### Sample Data", X.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "MLP (Neural Net)": MLPClassifier(max_iter=500)
}

# Train all models, get accuracies
accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

# Identify the best model
best_model_name = max(accuracies, key=accuracies.get)
display_names = {name: name for name in models}
display_names[best_model_name] = "SLSTM (Best Model)"

# Dropdown for selection
selected_display = st.selectbox("Choose a model", list(display_names.values()))

# Map back to real model name
reverse_map = {v: k for k, v in display_names.items()}
actual_model_name = reverse_map[selected_display]
model = models[actual_model_name]

# Retrain selected model and evaluate
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Results
acc = accuracy_score(y_test, y_pred)
st.write(f"### Accuracy: {acc * 100:.2f}%")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)
