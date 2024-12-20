import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Define the ontology model class
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

# Streamlit app
st.title("Neural Net Ontology Model Deployment")

# Sidebar for loading the model
st.sidebar.header("Model Setup")
model_file = st.sidebar.file_uploader("Upload Trained Model (.pth)", type=["pth"])

# Define constants
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 5

# Load the model if file is uploaded
model = None
if model_file is not None:
    model = OntologyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    try:
        model.load_state_dict(torch.load(model_file))
        model.eval()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# Sidebar for user input
st.sidebar.header("Input Features")
input_features = []

for i in range(INPUT_SIZE):
    value = st.sidebar.number_input(f"Feature {i+1}", value=0.0, step=0.1)
    input_features.append(value)

# Predict button
if st.button("Predict"):
    if model is None:
        st.error("Please upload a trained model first.")
    else:
        try:
            input_data = np.array(input_features).reshape(1, -1)
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                prediction = model(input_tensor).numpy()
            st.write("Prediction:", prediction)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Display model architecture
if st.checkbox("Show Model Architecture"):
    if model is None:
        st.warning("Model not loaded yet.")
    else:
        st.write(model)

# Debugging Info (optional)
if st.checkbox("Show Debug Info"):
    st.write("Input Features:", input_features)
