import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import io

st.set_page_config(page_title="Interactive Passport Crop", layout="centered")
st.title("Interactive Passport Photo Crop Tool with Live Rectangle Overlay")

uploaded = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])

# ---------------- Helpers ----------------
def resize_large(img, max_dim=1200):
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        return img.resize((new_w,new_h), Image.LANCZOS)
    return img

def add_border(img, color=(120,120,120), width=1):
    draw = ImageDraw.Draw(img)
    w,h = img.size
    draw.rectangle([0.5,0.5,w-0.5,h-0.5], outline=color, width=width)
    return img

def apply_crop_with_white(img, left, top, right, bottom):
    w, h = img.size
    # Create white background
    bg = Image.new("RGB", (w,h), "white")
    # Paste the cropped area
    crop_box = img.crop((left, top, right, bottom))
    bg.paste(crop_box, (left, top))
    return bg.crop((left, top, right, bottom))

def draw_overlay(img, left, top, right, bottom):
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([left, top, right, bottom], outline=(255,0,0), width=3)
    return overlay

# ---------------- Main ----------------
if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    original = resize_large(original)
    w, h = original.size

    st.subheader("Original Photo with Crop Overlay")

    # Initialize sliders
    col1, col2 = st.columns(2)

    with col1:
        left = st.slider("Left", 0, w-1, 0)
        top = st.slider("Top", 0, h-1, 0)
        right = st.slider("Right", left+1, w, w)
        bottom = st.slider("Bottom", top+1, h, h)

    # Draw live rectangle overlay
    overlay_img = draw_overlay(original, left, top, right, bottom)
    st.image(overlay_img, use_column_width=True)

    # Apply crop with white background
    cropped_with_bg = apply_crop_with_white(original, left, top, right, bottom)
    # Resize to passport size
    final = cropped_with_bg.resize((630,810), Image.LANCZOS)
    final = add_border(final)

    with col2:
        st.subheader("Cropped Passport Photo (630×810 px)")
        st.image(final, use_column_width=True)
        buf = io.BytesIO()
        final.save(buf, "JPEG", quality=95)
        st.download_button("Download Cropped Photo", buf.getvalue(), file_name="passport_manual_630x810.jpg", mime="image/jpeg")
