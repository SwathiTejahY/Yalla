
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# For SLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping

st.title("Simple Model Comparison with SLSTM")

# Load data
dftrain = pd.read_csv("train_small.csv")
dftest = pd.read_csv("test_small.csv")

# Encode if categorical
for df in [dftrain, dftest]:
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

x_train = dftrain.drop("target", axis=1).values
y_train = dftrain["target"].values
x_test = dftest.drop("target", axis=1).values
y_test = dftest["target"].values

# Scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Classical ML Models
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})

# SLSTM Model
st.write("Training SLSTM model...")

x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

slstm_model = Sequential()
slstm_model.add(Conv1D(64, 2, activation='relu', input_shape=(x_train.shape[1], 1)))
slstm_model.add(MaxPooling1D(2))
slstm_model.add(LSTM(50, return_sequences=True))
slstm_model.add(LSTM(50))
slstm_model.add(Dense(len(np.unique(y_train)), activation='softmax'))
slstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

monitor = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
slstm_model.fit(x_train_lstm, y_train, validation_data=(x_test_lstm, y_test),
                callbacks=[monitor], epochs=10, batch_size=64, verbose=0)

y_pred_slstm = np.argmax(slstm_model.predict(x_test_lstm), axis=-1)
acc = accuracy_score(y_test, y_pred_slstm)
f1 = f1_score(y_test, y_pred_slstm, average='weighted')
results.append({"Model": "SLSTM", "Accuracy": acc, "F1 Score": f1})

# Show results
results_df = pd.DataFrame(results)
st.dataframe(results_df)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, ax=ax[0])
ax[0].set_title("Accuracy")
sns.barplot(x="Model", y="F1 Score", data=results_df, ax=ax[1])
ax[1].set_title("F1 Score")
st.pyplot(fig)
