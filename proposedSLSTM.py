import streamlit as st
from PIL import Image
import time
import os

st.set_page_config(layout="wide")
st.title("📘 SLSTM Architecture - Animated Explanation")

# Step-by-step explanation paired with image filenames
steps = [
    ("Step 1: Input Sequence → First LSTM Layer", "step1.png"),
    ("Step 2: First LSTM → Second LSTM Layer", "step2.png"),
    ("Step 3: Second LSTM → Dense Layers", "step3.png"),
    ("Step 4: Dense Layers → Output Layer", "step4.png"),
    ("Step 5: Forget Gate Logic", "step5.png"),
    ("Step 6: Input Gate and Candidate Memory", "step6.png"),
    ("Step 7: Output Gate and Final Hidden State", "step7.png"),
]

# Add autoplay animation
if st.button("▶ Play SLSTM Animation"):
    for title, filename in steps:
        st.subheader(title)
        if os.path.exists(filename):
            image = Image.open(filename)
            st.image(image, use_column_width=True)
            time.sleep(2)
        else:
            st.error(f"Missing image file: {filename}")
        st.markdown("---")

# Static view as fallback
st.markdown("### 📚 View Slides Manually")
selected_step = st.selectbox("Choose Step", options=[title for title, _ in steps])
for title, filename in steps:
    if title == selected_step:
        st.subheader(title)
        if os.path.exists(filename):
            st.image(Image.open(filename), use_column_width=True)
        else:
            st.warning("Image not found.")
        break

st.markdown("---")
st.info("This animation illustrates how SLSTM manages memory with forget, input, and output gates to generate a final prediction.")
