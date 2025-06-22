import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
st.title("Multi-Classifier App (with SLSTM)")
import sys
st.write("Python version:", sys.version)

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

# User selects model
model_name = st.selectbox("Choose a model", [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "MLP (Neural Net)",
    "SLSTM (Stacked LSTM)"
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_name != "SLSTM (Stacked LSTM)":
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

    model = models[model_name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

else:
    # SLSTM: reshape and use TensorFlow
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = np.expand_dims(X_train.values, axis=1)
    X_test_lstm = np.expand_dims(X_test.values, axis=1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(1, X.shape[1])))
    model.add(LSTM(32))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_lstm, y_train_cat, epochs=30, batch_size=8, verbose=0)

    y_pred_probs = model.predict(X_test_lstm)
    y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
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
