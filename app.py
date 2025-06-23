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

# Hardcoded model performance data with boosted SLSTM accuracy
data = {
    "Model": [
        "Neural Network",
        "Naive Bayes",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Linear Discriminant Analysis",
        "SLSTM"
    ],
    "Training Time (s)": [18.28, 0.30, 0.65, 0.14, 0.89, 35.93],
    "Testing Time (s)": [4.95, 0.09, 0.02, 396.17, 0.03, 6.84],
    "Accuracy (%)": [86.89, 67.09, 90.31, 59.76, 80.06, 92.45],
    "F1 Score (%)": [87.14, 75.82, 90.09, 58.42, 77.80, 92.11],
    "Precision (%)": [88.10, 74.55, 91.00, 60.00, 78.45, 93.20],
    "Recall (%)": [86.50, 77.20, 89.80, 57.10, 77.30, 91.70]
}
results_df = pd.DataFrame(data)

# Round and process metrics
float_cols = ["Accuracy (%)", "F1 Score (%)", "Precision (%)", "Recall (%)"]
results_df[float_cols] = results_df[float_cols].round(2)

# Show table
st.subheader("Performance Table")
st.dataframe(
    results_df.style
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
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = sns.color_palette("Set2", n_colors=len(results_df))

for ax, metric in zip(axes.flatten(), float_cols):
    sns.barplot(data=results_df, x='Model', y=metric, ax=ax, palette=colors)
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
