import streamlit as st
from PIL import Image
import time

st.title("📊 SLSTM Architecture Explanation")

# Load images (you need to create and upload these 7 stages as separate images)
steps = [
    ("Step 1: Input Sequence to 1st LSTM Layer", "step1.png"),
    ("Step 2: 1st to 2nd LSTM Layer", "step2.png"),
    ("Step 3: LSTM to Dense Layers", "step3.png"),
    ("Step 4: Final Output Layer", "step4.png"),
    ("Step 5: Forget Gate", "step5.png"),
    ("Step 6: Input Gate", "step6.png"),
    ("Step 7: Output Gate & Hidden State", "step7.png")
]

if st.button("▶ Play Animation"):
    for title, path in steps:
        st.subheader(title)
        st.image(Image.open(path), use_column_width=True)
        time.sleep(2)

st.markdown("---")
st.info("Each gate (forget, input, output) manages part of the memory update using trainable weights and activation functions.")
