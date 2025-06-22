from sklearn.neural_network import MLPClassifier
# coding: utf-8


import time

def measure_time(model, x_train, y_train, x_test, y_test):
    start_train = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = model.predict(x_test)
    test_time = time.time() - start_test

    return y_pred, train_time, test_time

# In[ ]:


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
from sklearn.exceptions import ConvergenceWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:




# In[ ]:


dftrain = pd.read_csv("train_small.csv")
dftest = pd.read_csv("test_small.csv")


# In[ ]:


# Function to preprocess data
def preprocess_data(df):
    df = df.astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df


# In[ ]:


# Preprocess training data
dftrain = preprocess_data(dftrain)
x_train = dftrain.drop('target', axis=1).values
y_train = dftrain['target'].values


# In[ ]:


# Preprocess test data
dftest = preprocess_data(dftest)
x_test = dftest.drop('target', axis=1).values
y_test = dftest['target'].values


# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
X_train_scaled = scaler.transform(x_train)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_test)
X_test_scaled = scaler.transform(x_test)


# In[ ]:


x_test_normalized = (x_test - x_test.mean()) / x_test.std()

print("Original Data:")
print(x_test)

print("\nZ-score Normalized Data:")
print(x_test_normalized)


# In[ ]:


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


# In[ ]:


def evaluate_model(name, y_test, y_pred, train_time, test_time):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{name} - Training time: {train_time:.2f}s, Test time: {test_time:.2f}s, Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    return {'Model': name, 'Training Time': train_time, 'Testing Time': test_time, 'Accuracy': accuracy, 'F1 Score': f1}


# In[ ]:


results = []


# In[ ]:


print("Starting Neural Network")
# Removed keras add layers; using MLPClassifier instead(Dense(50, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
# Removed keras add layers; using MLPClassifier instead(Dense(30, activation='relu'))
# Removed keras add layers; using MLPClassifier instead(Dense(20, kernel_initializer='normal', activation='relu'))
# Removed keras add layers; using MLPClassifier instead(Dense(6, activation='softmax'))
# Removed keras compile for sklearn MLPClassifierloss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
nn_model.fit(x_train, y_train)  # Updated to scikit-learn format


# In[ ]:


# Measure prediction and timing
y_pred_nn, nn_train_time, nn_test_time = measure_time(nn_model, x_train, y_train, x_test, y_test)

# Convert probabilities to class predictions
import numpy as np

# Evaluate and append results
results.append(evaluate_model("Neural Network", y_test, y_pred_nn, nn_train_time, nn_test_time))




# In[ ]:


from sklearn.naive_bayes import GaussianNB
import time

# Define measure_time if not already defined
def measure_time(model, x_train, y_train, x_test, y_test):
    start_train = time.time()
    model.fit(x_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(x_test)
    end_test = time.time()

    train_time = end_train - start_train
    test_time = end_test - start_test

    return y_pred, train_time, test_time

# Run Naive Bayes model
print("Starting Naive Bayes")
gnb = GaussianNB()
y_pred_nb, nb_train_time, nb_test_time = measure_time(gnb, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Naive Bayes", y_test, y_pred_nb, nb_train_time, nb_test_time))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
import time

# Define the measure_time function if not already defined
def measure_time(model, x_train, y_train, x_test, y_test):
    start_train = time.time()
    model.fit(x_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred = model.predict(x_test)
    end_test = time.time()

    train_time = end_train - start_train
    test_time = end_test - start_test

    return y_pred, train_time, test_time

# Run Decision Tree model
print("Starting Decision Tree")
clf = DecisionTreeClassifier()
y_pred_dt, dt_train_time, dt_test_time = measure_time(clf, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Decision Tree", y_test, y_pred_dt, dt_train_time, dt_test_time))


# In[ ]:


print("Starting K-Nearest Neighbors")
knn = KNeighborsClassifier(n_neighbors=5)
y_pred_knn, knn_train_time, knn_test_time = measure_time(knn, x_train, y_train, x_test, y_test)
results.append(evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn, knn_train_time, knn_test_time))


# In[ ]:


print("Starting Linear Discriminant Analysis")
lda = LinearDiscriminantAnalysis()
y_pred_lda, lda_train_time, lda_test_time = measure_time(lda, x_train, y_train, x_test, y_test)
results.append(evaluate_model("Linear Discriminant Analysis", y_test, y_pred_lda, lda_train_time, lda_test_time))


# In[ ]:


print("Starting Convolutional Neural Network")
## Removed keras add layers; using MLPClassifier instead(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)))
## Removed keras add layers; using MLPClassifier instead(MaxPooling1D(pool_size=2))
## Removed keras add layers; using MLPClassifier instead(Flatten())
## Removed keras add layers; using MLPClassifier instead(Dense(50, activation='relu'))
## Removed keras add layers; using MLPClassifier instead(Dense(6, activation='softmax'))
## Removed keras compile for sklearn MLPClassifierloss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


x_train_cnn = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

from sklearn.neural_network import MLPClassifier
cnn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

cnn_model.fit(x_train_cnn, y_train)


# In[2]:


import numpy as np
import time
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Step 2: Define CNN model
def cnn_model():
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Step 3: Measure training and prediction time
def measure_time(model_func, x_train, y_train, x_test, y_test):
    model = model_func()

    start_train = time.time()
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    end_train = time.time()

    start_test = time.ti_



# In[5]:


import numpy as np
import time
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 2: Reshape for Conv1D + LSTM input (samples, timesteps, features)
x_train_slstm = x_train.reshape(-1, 28, 28)  # 28 time steps, 28 features each
x_test_slstm = x_test.reshape(-1, 28, 28)

slstm_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(28, 28)))
slstm_model.add(MaxPooling1D(pool_size=2))
slstm_model.add(LSTM(50, return_sequences=True))
slstm_model.add(LSTM(50))
slstm_model.add(Dense(10, activation='softmax'))  # 10 output classes for MNIST

slstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
start_train = time.time()
slstm_model.fit(x_train_slstm, y_train, epochs=3, batch_size=128, verbose=1)
end_train = time.time()

# Step 5: Make predictions
start_test = time.time()
y_pred_probs = slstm_model.predict(x_test_slstm)
end_test = time.time()


# Step 6: Evaluate performance
acc = accuracy_score(y_test, y_pred_slstm)
print(f"Training Time: {end_train - start_train:.2f} seconds")
print(f"Testing Time: {end_test - start_test:.2f} seconds")


# In[ ]:


slstm_history = slstm_model.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),  verbose=2, epochs=200, batch_size=1000)


# In[7]:


import numpy as np
import time
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def slstm_model():
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(28, 28)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(10, activation='softmax'))  # 10 classes for MNIST
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 3: Define the measure_time function
def measure_time(model_func, x_train, y_train, x_test, y_test, reshape=False):
    if reshape:
        x_train = x_train.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)

    model = model_func()

    start_train = time.time()
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    end_train = time.time()

    start_test = time.time()
    y_pred_probs = model.predict(x_test)
    end_test = time.time()


    return y_pred, end_train - start_train, end_test - start_test

# Step 4: Define the evaluation function
def evaluate_model(name, y_true, y_pred, train_time, test_time):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")

    return {
        "Model": name,
        "Accuracy": acc,
        "Train Time (s)": train_time,
        "Test Time (s)": test_time
    }

results = []

y_pred_slstm, slstm_train_time, slstm_test_time = measure_time(
    slstm_model, x_train, y_train, x_test, y_test, reshape=True
)

results.append(
)


# In[ ]:


results_df = pd.DataFrame(results)

# Display results
print("\nModel Performance Comparison:")
print(results_df)


# In[ ]:


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

