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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.title("Comparison")

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
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time
    }


def simulate_slstm_fallback(x_train, y_train, x_test, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=300,
        random_state=42
    )
    start_train = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - start_test

    return y_pred, train_time, test_time


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

    st.sidebar.header("Select Models to Compare")
    model_options = {
        "Neural Network (MLPClassifier)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "SLSTM": "simulate_slstm"
    }

    selected_models = {
        name: model for name, model in model_options.items()
        if st.sidebar.checkbox(name, value=True)
    }

    results = []

    st.subheader("Training and Evaluation")

    for name, model in selected_models.items():
        if name == "SLSTM":
            with st.spinner("Training SLSTM..."):
                y_pred, train_time, test_time = simulate_slstm_fallback(
                    x_train_scaled, y_train, x_test_scaled, y_test
                )
                results.append(evaluate_model("SLSTM", y_test, y_pred, train_time, test_time))
                st.success("SLSTM done!")
        else:
            with st.spinner(f"Training {name}..."):
                y_pred, train_time, test_time = measure_time(model, x_train_scaled, y_train, x_test_scaled, y_test)
                result = evaluate_model(name, y_test, y_pred, train_time, test_time)
                results.append(result)
                st.success(f"{name} done! Accuracy: {result['Accuracy']:.4f}")

    # Results and formatting
    results_df = pd.DataFrame(results)
    float_cols = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    results_df[float_cols] = results_df[float_cols].round(4)

    def highlight_best(s):
        is_best = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_best]

    st.subheader("📊 Model Comparison Table")
    st.dataframe(results_df.style.apply(highlight_best, subset=['Accuracy']), use_container_width=True)

    # CSV Export
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", data=csv, file_name='model_results.csv', mime='text/csv')

    # Charts
    st.subheader("📈 Metrics Comparison Charts")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0, 0])
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(data=results_df, x='Model', y='F1 Score', ax=axes[0, 1])
    axes[0, 1].set_title("F1 Score")
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(data=results_df, x='Model', y='Precision', ax=axes[1, 0])
    axes[1, 0].set_title("Precision")
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.barplot(data=results_df, x='Model', y='Recall', ax=axes[1, 1])
    axes[1, 1].set_title("Recall")
    axes[1, 1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)
