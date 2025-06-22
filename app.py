
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.title("Model Performance Comparison (with Time Metrics)")

@st.cache_data
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Target")
    target_names = data.target_names
    return X, y, target_names

X, y, target_names = load_data()

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

results = []

for name, model in models.items():
    start_train = time.time()
    model.fit(X_train_scaled, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(X_test_scaled)
    end_test = time.time()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        "Model": name,
        "Training Time": end_train - start_train,
        "Testing Time": end_test - start_test,
        "Accuracy": acc,
        "F1 Score": f1
    })

df_results = pd.DataFrame(results)
st.subheader("📋 Model Performance Comparison:")
st.dataframe(df_results)

# Accuracy Bar Plot
st.subheader("📊 Model Accuracy Comparison")
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=df_results, x="Accuracy", y="Model", ax=ax_acc)
st.pyplot(fig_acc)

# F1 Score Bar Plot
st.subheader("📊 Model F1 Score Comparison")
fig_f1, ax_f1 = plt.subplots()
sns.barplot(data=df_results, x="F1 Score", y="Model", ax=ax_f1)
st.pyplot(fig_f1)

# Time Comparisons
st.subheader("⏱️ Training Time Comparison")
fig_tt, ax_tt = plt.subplots()
sns.barplot(data=df_results, x="Training Time", y="Model", ax=ax_tt)
st.pyplot(fig_tt)

st.subheader("⏱️ Testing Time Comparison")
fig_et, ax_et = plt.subplots()
sns.barplot(data=df_results, x="Testing Time", y="Model", ax=ax_et)
st.pyplot(fig_et)
