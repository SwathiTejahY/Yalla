# app.py

import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam

st.title("SLSTM Model (Conv1D + LSTM) on Tabular Data")

train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Testing CSV", type="csv")


def preprocess_data(df):
    df = df.astype("category")
    cat_cols = df.select_dtypes(["category"]).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
    return df


def run_slstm(x_train, y_train, x_test, y_test):
    x_train_r = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_r = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential()
    model.add(Conv1D(64, 2, activation="relu", input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(len(np.unique(y_train)), activation="softmax"))
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

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

    dftrain = preprocess_data(dftrain)
    dftest = preprocess_data(dftest)

    x_train = dftrain.drop("target", axis=1).values
    y_train = dftrain["target"].values
    x_test = dftest.drop("target", axis=1).values
    y_test = dftest["target"].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    st.info("Training SLSTM...")
    y_pred, train_time, test_time = run_slstm(x_train, y_train, x_test, y_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    st.success("Training complete!")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**Training Time:** {train_time:.2f}s")
    st.write(f"**Testing Time:** {test_time:.2f}s")
