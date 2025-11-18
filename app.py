import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import io

st.set_page_config(page_title="Passport Photo Editor AI", layout="centered")
st.markdown("<h2 style='text-align:center; color:blue;'>Passport Photo Editor AI integrated - AlankitGlobel Germany</h2>", unsafe_allow_html=True)

# ---------------- Selection Menus ----------------
subject_type = st.selectbox("Select Subject Type", ["Man", "Woman", "Baby"])
beard_type = st.selectbox("Select Beard/Turban", ["Without Beard", "With Beard/Turban"])
photo_type = st.selectbox("Select Photo Type", ["Passport", "Visa/OCI"])
no_change = st.checkbox("No Changes to Photo")

uploaded = st.file_uploader("Upload your portrait photo", type=["jpg","jpeg","png"])

# ---------------- Helper Functions ----------------
def resize_large(img, max_dim=1200):
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        return img.resize((new_w,new_h), Image.LANCZOS)
    return img

def add_border(img, color=(120, 120, 120), width=1):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    draw.rectangle([0.5, 0.5, w-0.5, h-0.5], outline=color, width=width)
    return img

def flashy_background(img, color=(255,0,255)):
    """Simple flashy background by filling corners and edges"""
    arr = np.array(img)
    h, w = arr.shape[:2]
    bg_color = np.full_like(arr, color, dtype=np.uint8)
    # Create mask for center rectangle (keep original)
    mask = np.zeros((h,w), dtype=np.uint8)
    margin_w = w//3
    margin_h = h//3
    mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1
    mask = mask[:,:,np.newaxis]
    # Blend background
    blended = arr*mask + bg_color*(1-mask)
    return Image.fromarray(blended.astype(np.uint8))

def enhance(img):
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

# ---------------- Main ----------------
if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    original = resize_large(original)

    st.subheader("Original Photo")
    st.image(original, use_column_width=True)

    if no_change:
        st.subheader("No Changes Applied")
        final_img = original
    else:
        # Determine flashy color based on subject type
        color_map = {"Man": (0,255,255), "Woman": (255,105,180), "Baby": (255,255,0)}
        color = color_map.get(subject_type, (255,0,255))

        edited = flashy_background(original, color=color)
        final_img = enhance(edited)

    st.subheader("Edited Photo Preview")
    st.image(final_img, use_column_width=True)

    buf = io.BytesIO()
    final_img.save(buf, "JPEG", quality=95)
    st.download_button("Download Edited Photo", buf.getvalue(), file_name="edited_passport_photo.jpg", mime="image/jpeg")
