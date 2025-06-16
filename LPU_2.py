import streamlit as st

# Streamlit app based on converted notebook
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

dftrain = pd.read_csv("/content/drive/MyDrive/LPU4/train70_reduced.csv")
dftest = pd.read_csv("/content/drive/MyDrive/LPU4/test30_reduced.csv")

# Function to preprocess data
def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

# Preprocess training data
dftrain = preprocess_data(dftrain)
x_train = dftrain.drop('target', axis=1).values
y_train = dftrain['target'].values

# Preprocess test data
dftest = preprocess_data(dftest)
x_test = dftest.drop('target', axis=1).values
y_test = dftest['target'].values

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
X_train_scaled = scaler.transform(x_train)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_test)
X_test_scaled = scaler.transform(x_test)

x_test_normalized = (x_test - x_test.mean()) / x_test.std()

print("Original Data:")
print(x_test)

print("\nZ-score Normalized Data:")
print(x_test_normalized)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = X_test_scaled
y = y_test

# Initialize the model (estimator)
model = LogisticRegression(max_iter=500)

# Initialize RFE and select top 2 features
rfe = RFE(estimator=model, n_features_to_select=2)

# Fit RFE
rfe.fit(X, y)

# Get mask of selected features (True = selected)
print("Selected features mask:", rfe.support_)

# Get ranking of features (1 = best)
print("Feature ranking:", rfe.ranking_)

# Transform the data to selected features only
X_rfe = rfe.transform(X)
print("Shape of original data:", X.shape)
print("Shape after RFE:", X_rfe.shape)


def evaluate_model(name, y_test, y_pred, train_time, test_time):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - Training time: {train_time:.2f}s, Test time: {test_time:.2f}s, Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    return {'Model': name, 'Training Time': train_time, 'Testing Time': test_time, 'Accuracy': accuracy, 'F1 Score': f1}

results = []

print("Starting Neural Network")
nn_model = Sequential()
nn_model.add(Dense(50, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
nn_model.add(Dense(30, activation='relu'))
nn_model.add(Dense(20, kernel_initializer='normal', activation='relu'))
nn_model.add(Dense(6, activation='softmax'))
nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
history = nn_model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=200, batch_size=1000)

y_pred_nn, nn_train_time, nn_test_time = measure_time(nn_model, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Neural Network", y_test, y_pred_nn, nn_train_time, nn_test_time))

print("Starting Naive Bayes")
gnb = GaussianNB()
y_pred_nb, nb_train_time, nb_test_time = measure_time(gnb, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Naive Bayes", y_test, y_pred_nb, nb_train_time, nb_test_time))

print("Starting Decision Tree")
clf = DecisionTreeClassifier()
y_pred_dt, dt_train_time, dt_test_time = measure_time(clf, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Decision Tree", y_test, y_pred_dt, dt_train_time, dt_test_time))

print("Starting K-Nearest Neighbors")
knn = KNeighborsClassifier(n_neighbors=5)
y_pred_knn, knn_train_time, knn_test_time = measure_time(knn, x_train, y_train, x_test, y_test)
results.append(evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn, knn_train_time, knn_test_time))

print("Starting Linear Discriminant Analysis")
lda = LinearDiscriminantAnalysis()
y_pred_lda, lda_train_time, lda_test_time = measure_time(lda, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Linear Discriminant Analysis", y_test, y_pred_lda, lda_train_time, lda_test_time))

print("Starting Convolutional Neural Network")
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(6, activation='softmax'))
cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train_cnn = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
cnn_history = cnn_model.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test), callbacks=[monitor], verbose=2, epochs=200, batch_size=1000)

y_pred_cnn, cnn_train_time, cnn_test_time = measure_time(cnn_model, x_train, y_train, x_test, y_test, reshape=True)
results.append(evaluate_model("Convolutional Neural Network", y_test, y_pred_cnn, cnn_train_time, cnn_test_time))

print("Starting SLSTM")
slstm_model = Sequential()
slstm_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)))
slstm_model.add(MaxPooling1D(pool_size=2))
slstm_model.add(LSTM(50, return_sequences=True))
slstm_model.add(LSTM(50))
slstm_model.add(Dense(6, activation='softmax'))
slstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

slstm_history = slstm_model.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test), callbacks=[monitor], verbose=2, epochs=200, batch_size=1000)

y_pred_slstm, slstm_train_time, slstm_test_time = measure_time(slstm_model, x_train, y_train, x_test, y_test, reshape=True)
results.append(evaluate_model("SLSTM", y_test, y_pred_slstm, slstm_train_time, slstm_test_time))

results_df = pd.DataFrame(results)

# Display results
print("\nModel Performance Comparison:")
print(results_df)

# Plot accuracy and F1 score including all models
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title('Model F1 Score Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()