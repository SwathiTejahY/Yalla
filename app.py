
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Model Performance Comparison App (Accuracy & F1 Score)")
st.write("🚨 Python version in use:", sys.version)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=500)
}

accuracies = []
f1_scores = []
model_names = []
reports = {}
cms = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracies.append(acc)
    f1_scores.append(f1)
    model_names.append(name)
    reports[name] = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cms[name] = confusion_matrix(y_test, y_pred)

# Accuracy bar chart
st.subheader("📊 Model Accuracy Comparison")
fig1, ax1 = plt.subplots()
sns.barplot(x=accuracies, y=model_names, ax=ax1)
ax1.set_xlabel("Accuracy")
ax1.set_ylabel("Model")
st.pyplot(fig1)

# F1 score bar chart
st.subheader("📊 Model F1 Score Comparison")
fig2, ax2 = plt.subplots()
sns.barplot(x=f1_scores, y=model_names, ax=ax2)
ax2.set_xlabel("F1 Score")
ax2.set_ylabel("Model")
st.pyplot(fig2)

# Optional detailed view
selected_model = st.selectbox("🔍 Select a model to inspect", model_names)
st.write(f"### Classification Report: {selected_model}")
st.dataframe(pd.DataFrame(reports[selected_model]).T)

st.write(f"### Confusion Matrix: {selected_model}")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cms[selected_model], annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig_cm)
