import streamlit as st
from PIL import Image
import time

st.set_page_config(layout="wide")
st.title("📘 SLSTM Architecture - Animated Explanation")

# Upload all 7 step images at once
uploaded_images = st.file_uploader(
    "📂 Upload All SLSTM Step Images (step1.png to step7.png)",
    type=["png"],
    accept_multiple_files=True,
    help="Upload all 7 PNG images named step1.png through step7.png"
)

# Step-by-step descriptions and filenames
steps = [
    ("Step 1: Input Sequence → First LSTM Layer", "step1.png", "The input sequence is passed to the first LSTM layer, which begins the temporal processing of the sequence data."),
    ("Step 2: First LSTM → Second LSTM Layer", "step2.png", "The first LSTM's output is passed to the second stacked LSTM layer for deeper sequence learning."),
    ("Step 3: Second LSTM → Dense Layers", "step3.png", "Outputs from the second LSTM are connected to dense (fully connected) layers to map learned features to the output domain."),
    ("Step 4: Dense Layers → Output Layer", "step4.png", "The final dense layer computes the network output, typically used for classification or regression."),
    ("Step 5: Forget Gate Logic", "step5.png", "The forget gate controls which parts of the previous cell state should be retained or discarded using a sigmoid filter."),
    ("Step 6: Input Gate and Candidate Memory", "step6.png", "The input gate updates memory content with new candidate values generated using tanh and filtered by a sigmoid gate."),
    ("Step 7: Output Gate and Final Hidden State", "step7.png", "The output gate determines the final hidden state using the updated memory and sigmoid/tanh operations."),
]

# Map images by filename
image_dict = {}
if uploaded_images:
    for img in uploaded_images:
        image_dict[img.name.lower()] = Image.open(img)

# Animation all at once with explanations
if st.button("▶ View Full SLSTM Animation"):
    for title, filename, explanation in steps:
        st.subheader(title)
        st.markdown(f"**Explanation:** {explanation}")
        if filename.lower() in image_dict:
            st.image(image_dict[filename.lower()], use_column_width=True)
        else:
            st.error(f"❌ Missing image: {filename}")
        st.markdown("---")

st.markdown("---")
st.info("Upload all SLSTM step visuals (step1.png to step7.png) to view a complete architectural animation with step-by-step explanation.")
