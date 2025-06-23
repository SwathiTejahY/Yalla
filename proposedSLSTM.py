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

# Expected order and filenames
steps = [
    ("Step 1: Input Sequence → First LSTM Layer", "step1.png"),
    ("Step 2: First LSTM → Second LSTM Layer", "step2.png"),
    ("Step 3: Second LSTM → Dense Layers", "step3.png"),
    ("Step 4: Dense Layers → Output Layer", "step4.png"),
    ("Step 5: Forget Gate Logic", "step5.png"),
    ("Step 6: Input Gate and Candidate Memory", "step6.png"),
    ("Step 7: Output Gate and Final Hidden State", "step7.png"),
]

# Map images by filename
image_dict = {}
if uploaded_images:
    for img in uploaded_images:
        image_dict[img.name.lower()] = Image.open(img)

# Animation
if st.button("▶ Play SLSTM Animation"):
    for title, filename in steps:
        st.subheader(title)
        if filename.lower() in image_dict:
            st.image(image_dict[filename.lower()], use_column_width=True)
            time.sleep(2)
        else:
            st.error(f"❌ Missing image: {filename}")
        st.markdown("---")

# Manual step viewer
st.markdown("### 📚 View Slides Manually")
selected_step = st.selectbox("Select Step", [title for title, _ in steps])
for title, filename in steps:
    if title == selected_step:
        st.subheader(title)
        if filename.lower() in image_dict:
            st.image(image_dict[filename.lower()], use_column_width=True)
        else:
            st.warning(f"⚠️ '{filename}' not uploaded.")
        break

st.markdown("---")
st.info("Upload all SLSTM step visuals, then play the animation or explore each step manually.")
