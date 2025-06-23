# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import time

st.title("MLPClassifier Model on Tabular Data")

# Upload training and test CSV files
train_file = st.file_uploader("Upload Train CSV", type="csv")
test_file = st.file_uploader("Upload Test CSV", type="csv")

def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

if train_file and test_file:
    dftrain = pd.read_csv(train_file)
    dftest = pd.read_csv(test_file)

    # Preprocess
    dftrain = preprocess_data(dftrain)
    dftest = preprocess_data(dftest)

    x_train = dftrain.drop('target', axis=1).values
    y_train = dftrain['target'].values
    x_test = dftest.drop('target', axis=1).values
    y_test = dftest['target'].values

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Flatten
    x_train_flat = x_train_scaled.reshape((x_train_scaled.shape[0], -1))
    x_test_flat = x_test_scaled.reshape((x_test_scaled.shape[0], -1))

    # Train model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    start_train = time.time()
    model.fit(x_train_flat, y_train)
    train_time = time.time() - start_train

    # Predict
    start_test = time.time()
    y_pred = model.predict(x_test_flat)
    test_time = time.time() - start_test

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Output
    st.success("Model trained and evaluated successfully!")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**Training Time:** {train_time:.2f} seconds")
    st.write(f"**Testing Time:** {test_time:.2f} seconds")

    # Show confusion matrix
    if st.checkbox("Show Confusion Matrix"):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
