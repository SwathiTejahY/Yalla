
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.title("Model Performance Summary (Formatted View)")

@st.cache_data
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")
    return X, y

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Neural Network": MLPClassifier(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
}

results = {
    "Model": [],
    "Training Time": [],
    "Testing Time": [],
    "Accuracy": [],
    "F1 Score": []
}

for name, model in models.items():
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(X_test_scaled)
    end_test = time.time()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results["Model"].append(name)
    results["Training Time"].append(round(end_train - start_train, 6))
    results["Testing Time"].append(round(end_test - start_test, 6))
    results["Accuracy"].append(round(acc, 6))
    results["F1 Score"].append(round(f1, 6))

df_summary = pd.DataFrame(results)

st.subheader("📋 Model Performance Comparison (Formatted):")
st.dataframe(df_summary)

st.subheader("📊 Accuracy Comparison")
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=df_summary, x="Accuracy", y="Model", ax=ax_acc)
st.pyplot(fig_acc)

st.subheader("📊 F1 Score Comparison")
fig_f1, ax_f1 = plt.subplots()
sns.barplot(data=df_summary, x="F1 Score", y="Model", ax=ax_f1)
st.pyplot(fig_f1)
