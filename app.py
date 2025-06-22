
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

st.title("Multi-Classifier Comparison App")
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
    "MLP (Neural Net)": MLPClassifier(max_iter=500)
}

accuracies = {}
reports = {}
cms = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    reports[name] = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cms[name] = confusion_matrix(y_test, y_pred)

# Display all accuracies in a bar chart
st.write("### Model Accuracy Comparison")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
fig_acc, ax_acc = plt.subplots()
sns.barplot(data=acc_df, x="Accuracy", y="Model", palette="viridis", ax=ax_acc)
st.pyplot(fig_acc)

# Optional model selection
selected_model = st.selectbox("Select a model to inspect", acc_df["Model"].tolist())

st.write(f"### Classification Report for {selected_model}")
st.dataframe(pd.DataFrame(reports[selected_model]).T)

st.write(f"### Confusion Matrix for {selected_model}")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cms[selected_model], annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig_cm)
