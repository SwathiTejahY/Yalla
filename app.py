import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Model Performance Comparison")

# Hardcoded model performance data
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
    "F1 Score (%)": [87.1373, 75.8161, 90.0899, 58.4157, 77.8011, 88.129],
    "Precision (%)": [88.25, 72.44, 91.12, 60.78, 76.42, 87.65],
    "Recall (%)": [85.94, 78.32, 89.74, 56.30, 79.92, 88.70]
}

# Create DataFrame
results_df = pd.DataFrame(data)
float_cols = ["Accuracy (%)", "F1 Score (%)", "Precision (%)", "Recall (%)"]
results_df[float_cols] = results_df[float_cols].round(2)

# Show table
st.subheader("Performance Table")
st.dataframe(
    results_df.style.highlight_max(axis=0, subset=["Accuracy (%)"]).format("{:.2f}"),
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

for ax, metric in zip(axes.flatten(), float_cols):
    sns.barplot(data=results_df, x='Model', y=metric, ax=ax)
    ax.set_title(metric)
    ax.set_ylim(0, 100)
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=45)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", label_type="edge", padding=2)

st.pyplot(fig)
