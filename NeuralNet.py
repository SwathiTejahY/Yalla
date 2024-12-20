import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Define your ontology model (example neural network)
class OntologyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OntologyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 5

model = OntologyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# Function to make predictions
def predict(input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return output.numpy()

# Streamlit app
st.title("Neural Net Ontology Model Deployment")

# Sidebar for user input
st.sidebar.header("Input Features")
input_features = []

for i in range(INPUT_SIZE):
    value = st.sidebar.number_input(f"Feature {i+1}", value=0.0, step=0.1)
    input_features.append(value)

# Predict button
if st.button("Predict"):
    input_data = np.array(input_features).reshape(1, -1)
    prediction = predict(input_data)
    st.write("Prediction:", prediction)

# Display model architecture
if st.checkbox("Show Model Architecture"):
    st.write(model)
