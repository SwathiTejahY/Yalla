import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Fast Model Comparison App")

# ===================== Load & Preprocess =====================
@st.cache_data
def load_data():
    dftrain = pd.read_csv("train70_reduced.csv")
    dftest = pd.read_csv("test30_reduced.csv")

    dftrain = dftrain.astype('category')
    dftest = dftest.astype('category')

    for df in [dftrain, dftest]:
        cat_cols = df.select_dtypes(['category']).columns
        df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)

    x_train = dftrain.drop('target', axis=1).values
    y_train = dftrain['target'].values
    x_test = dftest.drop('target', axis=1).values
    y_test = dftest['target'].values

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# ===================== RFE Feature Selection =====================
model = LogisticRegression(max_iter=500)
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(x_test, y_test)
st.write("### RFE Feature Selection")
st.write("Selected Features Mask:", rfe.support_)
st.write("Feature Ranking:", rfe.ranking_)

# ===================== Model Utils =====================
def measure_time(model, x_train, y_train, x_test, y_test, reshape=False):
    if reshape:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    start_train = time.time()
    model.fit(x_train, y_train)
    end_train = time.time()
    start_test = time.time()
    y_pred = model.predict(x_test)
    end_test = time.time()
    return y_pred, end_train - start_train, end_test - start_test

def evaluate_model(name, y_test, y_pred, train_time, test_time):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    st.write(f"**{name}** - Train: {train_time:.2f}s, Test: {test_time:.2f}s, Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return {'Model': name, 'Training Time': train_time, 'Testing Time': test_time, 'Accuracy': acc, 'F1 Score': f1}

results = []

# ===================== Classical ML Models =====================
st.subheader("Classical Machine Learning Models")

with st.spinner("Running Naive Bayes..."):
    gnb = GaussianNB()
    y_pred, t_train, t_test = measure_time(gnb, x_train, y_train, x_test, y_test)
    results.append(evaluate_model("Naive Bayes", y_test, y_pred, t_train, t_test))

with st.spinner("Running Decision Tree..."):
    clf = DecisionTreeClassifier()
    y_pred, t_train, t_test = measure_time(clf, x_train, y_train, x_test, y_test)
    results.append(evaluate_model("Decision Tree", y_test, y_pred, t_train, t_test))

with st.spinner("Running K-Nearest Neighbors..."):
    knn = KNeighborsClassifier(n_neighbors=5)
    y_pred, t_train, t_test = measure_time(knn, x_train, y_train, x_test, y_test)
    results.append(evaluate_model("K-Nearest Neighbors", y_test, y_pred, t_train, t_test))

with st.spinner("Running Linear Discriminant Analysis..."):
    lda = LinearDiscriminantAnalysis()
    y_pred, t_train, t_test = measure_time(lda, x_train, y_train, x_test, y_test)
    results.append(evaluate_model("LDA", y_test, y_pred, t_train, t_test))

# ===================== Deep Learning (Optional) =====================
st.subheader("Deep Learning Models (Optional)")
run_dl = st.checkbox("Run Deep Learning Models (Slow)", value=False)

if run_dl:
    monitor = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

    # Use subset for validation to reduce load
    x_val = x_test[:300]
    y_val = y_test[:300]

    with st.spinner("Training Neural Network..."):
        nn_model = Sequential([
            Dense(50, input_dim=x_train.shape[1], activation='relu'),
            Dense(30, activation='relu'),
            Dense(20, activation='relu'),
            Dense(6, activation='softmax')
        ])
        nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        nn_model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[monitor]

