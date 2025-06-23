# app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


st.title("Compare ML Models on Tabular Data")

train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Testing CSV", type="csv")


def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


def measure_time(model, x_train, y_train, x_test, y_test):
    start_train = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - start_test

    return y_pred, train_time, test_time


def evaluate_model(name, y_true, y_pred, train_time, test_time):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {
        'Model': name,
        'Accuracy': acc,
        'F1 Score': f1,
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time
    }


if train_file and test_file:
    dftrain = pd.read_csv(train_file)
    dftest = pd.read_csv(test_file)

    st.subheader("Data Preview")
    st.write("Train Data", dftrain.head())
    st.write("Test Data", dftest.head())

    dftrain = preprocess_data(dftrain)
    dftest = preprocess_data(dftest)

    x_train = dftrain.drop('target', axis=1).values
    y_train = dftrain['target'].values
    x_test = dftest.drop('target', axis=1).values
    y_test = dftest['target'].values

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Neural Network (MLPClassifier)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
    }

    results = []

    st.subheader("Training and Evaluation")

    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            y_pred, train_time, test_time = measure_time(model, x_train_scaled, y_train, x_test_scaled, y_test)
            result = evaluate_model(name, y_test, y_pred, train_time, test_time)
            results.append(result)
            st.success(f"{name} done! Accuracy: {result['Accuracy']:.4f}, F1 Score: {result['F1 Score']:.4f}")

    results_df = pd.DataFrame(results)
    st.subheader("📊 Model Comparison Table")
    st.dataframe(results_df)

    # Plot results
    st.subheader("📈 Accuracy and F1 Score Chart")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=results_df, x='Model', y='Accuracy', ax=ax[0])
    ax[0].set_title("Accuracy")
    ax[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=results_df, x='Model', y='F1 Score', ax=ax[1])
    ax[1].set_title("F1 Score")
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)
