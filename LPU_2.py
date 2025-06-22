import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ===================== Streamlit Title =====================
st.title("Model Comparison on Classification Task")

# ===================== Load Data =====================
@st.cache_data
def load_data():
    train = pd.read_csv("train70_reduced.csv")
    test = pd.read_csv("test30_reduced.csv")
    return train, test

dftrain, dftest = load_data()

# ===================== Preprocessing =====================
def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

dftrain = preprocess_data(dftrain)
dftest = preprocess_data(dftest)

x_train = dftrain.drop('target', axis=1).values
y_train = dftrain['target'].values
x_test = dftest.drop('target', axis=1).values
y_test = dftest['target'].values

# Scaling
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ===================== RFE =====================
model = LogisticRegression(max_iter=500)
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(x_test_scaled, y_test)

# Display selected features
st.write("### RFE Feature Selection")
st.write("Selected Features Mask:", rfe.support_)
st.write("Feature Ranking:", rfe.ranking_)

# ===================== Helper Function =====================
def measure_time(model, x_train, y_train, x_test, y_test, reshape=False):
    if reshape:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    start_train = time.time()
    model.fit(x_train, y_train)
    end_train = time.time()
    
