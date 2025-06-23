import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Model Performance Comparison")

# CSV file uploader
uploaded_file = st.file_uploader("Upload CSV File with Model Performance Data", type=["csv"])

if uploaded_file:
    results_df = pd.read_csv(uploaded_file)

    # Required columns
    required_cols = [
        "Model",
        "Training Time (s)",
        "Testing Time (s)",
        "Accuracy (%)",
        "F1 Score (%)",
        "Precision (%)",
        "Recall (%)"
    ]

    if all(col in results_df.columns for col in required_cols):
        # Round values
        float_cols = ["Accuracy (%)", "F1 Score (%)", "Precision (%)", "Recall (%)"]
        results_df[float_cols] = results_df[float_cols].round(2)

        # Show performance table
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

    else:
        st.error("Uploaded CSV must contain the following columns:\n" + ", ".join(required_cols))
else:
    st.info("Please upload a CSV file containing model performance data.")
