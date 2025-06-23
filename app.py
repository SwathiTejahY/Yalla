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
        "Linear Discriminant Analysis"
    ],
    "Training Time (s)": [18.284866, 0.302774, 0.652905, 0.142416, 0.886611],
    "Testing Time (s)": [4.946178, 0.089319, 0.015780, 396.167002, 0.032304],
    "Accuracy (%)": [86.8929, 67.0863, 90.3122, 59.7613, 80.0564],
    "F1 Score (%)": [87.1373, 75.8161, 90.0899, 58.4157, 77.8011]
}

# Create DataFrame
results_df = pd.DataFrame(data)
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
