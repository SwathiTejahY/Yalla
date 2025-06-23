import streamlit as st
from PIL import Image
import time

st.set_page_config(layout="centered")
st.markdown("<h2 style='text-align: center; font-size: 24px;'>📘 SLSTM Architecture - Animated Explanation</h2>", unsafe_allow_html=True)

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

# Animated playback
if st.button("▶ Play Full Animation Step-by-Step"):
    for title, filename, explanation in steps:
        st.markdown(f"<h4 style='font-size:18px;'>{title}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;'><b>Explanation:</b> {explanation}</p>", unsafe_allow_html=True)
        if filename.lower() in image_dict:
            st.image(image_dict[filename.lower()], use_column_width=True)
            time.sleep(2)
        else:
            st.error(f"❌ Missing image: {filename}")
        st.markdown("---")

# Navigation control for single-step view
st.markdown("<h4 style='font-size:18px;'>🔄 Navigate SLSTM Steps</h4>", unsafe_allow_html=True)
step_titles = [title for title, _, _ in steps]
selected_step = st.selectbox("Choose Step to View", step_titles)

for title, filename, explanation in steps:
    if title == selected_step:
        st.markdown(f"<h4 style='font-size:18px;'>{title}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;'><b>Explanation:</b> {explanation}</p>", unsafe_allow_html=True)
        if filename.lower() in image_dict:
            st.image(image_dict[filename.lower()], use_column_width=True)
        else:
            st.error(f"❌ Missing image: {filename}")
        break

st.markdown("---")
st.markdown("<p style='font-size:13px;'>Upload all SLSTM step visuals (step1.png to step7.png) to view a complete architectural animation or navigate one step at a time.</p>", unsafe_allow_html=True)
