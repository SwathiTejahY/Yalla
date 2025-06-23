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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam


st.title("Compare ML Models on Tabular Data (with SLSTM)")

train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Testing CSV", type="csv")


def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


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


def run_slstm(x_train, y_train, x_test, y_test):
    # Reshape for Conv1D + LSTM
    x_train_r = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_r = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_train = time.time()
    model.fit(x_train_r, y_train, epochs=10, batch_size=32, verbose=0)
    train_time = time.time() - start_train

    start_test = time.time()
    y_probs = model.predict(x_test_r)
    y_pred = np.argmax(y_probs, axis=1)
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

    # Classic models
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
            start_train = time.time()
            model.fit(x_train_scaled, y_train)
            train_time = time.time() - start_train

            start_test = time.time()
            y_pred = model.predict(x_test_scaled)
            test_time = time.time() - start_test

            results.append(evaluate_model(name, y_test, y_pred, train_time, test_time))
            st.success(f"{name} done! Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Add SLSTM
    with st.spinner("Training SLSTM..."):
        y_pred_slstm, train_time, test_time = run_slstm(x_train_scaled, y_train, x_test_scaled, y_test)
        results.append(evaluate_model("SLSTM (Conv1D + LSTM)", y_test, y_pred_slstm, train_time, test_time))
        st.success("SLSTM done!")

    # Display results
    results_df = pd.DataFrame(results)
    st.subheader("📊 Model Comparison Table")
    st.dataframe(results_df)

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
