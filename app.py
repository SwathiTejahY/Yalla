import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Model Performance Comparison")

# CSV file uploaders
st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training CSV", type=["csv"], key="train")
test_file = st.sidebar.file_uploader("Upload Testing CSV", type=["csv"], key="test")
model_file = st.sidebar.file_uploader("Upload CSV File with Model Performance Data", type=["csv"], key="model")

if model_file:
    results_df = pd.read_csv(model_file)
else:
    # Fallback hardcoded model performance data including SLSTM
    data = {
        "Model": [
            "Neural Network",
            "Naive Bayes",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Linear Discriminant Analysis",
            "SLSTM"
        ],
        "Training Time (s)": [18.284866, 0.302774, 0.652905, 0.142416, 0.886611, 25.938],
        "Testing Time (s)": [4.946178, 0.089319, 0.015780, 396.167002, 0.032304, 5.341],
        "Accuracy (%)": [86.8929, 67.0863, 90.3122, 59.7613, 80.0564, 88.571],
        "F1 Score (%)": [87.1373, 75.8161, 90.0899, 58.4157, 77.8011, 88.129]
    }
    results_df = pd.DataFrame(data)

# Process and display
float_cols = ["Accuracy (%)", "F1 Score (%)"]
results_df[float_cols] = results_df[float_cols].round(2)

# Show table
st.subheader("Performance Table")
st.dataframe(
    results_df.style
        .highlight_max(axis=0, subset=["Accuracy (%)"])
        .format({col: "{:.2f}" for col in results_df.select_dtypes(include=['float']).columns}),
    use_container_width=True
)

# Display best model
best_model = results_df.loc[results_df["Accuracy (%)"].idxmax()]
st.markdown(f"### 🏆 Best Model: **{best_model['Model']}**")
st.markdown(f"**Accuracy:** {best_model['Accuracy (%)']:.2f}%")

# Download CSV
csv = results_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Results as CSV", data=csv, file_name='model_performance.csv', mime='text/csv')

# Charts
st.subheader("📈 Metrics Comparison Charts")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric in zip(axes, float_cols):
    sns.barplot(data=results_df, x='Model', y=metric, ax=ax)
    ax.set_title(metric)
    ax.set_ylim(0, 100)
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", label_type="edge", padding=2)

st.pyplot(fig)

# Optionally display uploaded training/testing data
if train_file:
    st.subheader("📁 Training Data Preview")
    train_df = pd.read_csv(train_file)
    st.dataframe(train_df.head())

if test_file:
    st.subheader("📁 Testing Data Preview")
    test_df = pd.read_csv(test_file)
    st.dataframe(test_df.head())
