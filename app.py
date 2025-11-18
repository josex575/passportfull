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

def flashy_background(img, subject_type):
    color_map = {"Man": (0,255,255), "Woman": (255,105,180), "Baby": (255,255,0)}
    color = color_map.get(subject_type, (255,0,255))
    arr = np.array(img)
    h, w = arr.shape[:2]
    bg_color = np.full_like(arr, color, dtype=np.uint8)
    # For women, preserve top 25% (hair) as original
    preserve = h//4 if subject_type=="Woman" else 0
    mask = np.zeros((h,w), dtype=np.uint8)
    mask[preserve:, :] = 1
    mask = mask[:,:,np.newaxis]
    blended = arr*mask + bg_color*(1-mask)
    return Image.fromarray(blended.astype(np.uint8))

def enhance(img):
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

def draw_overlay(img, left, top, right, bottom):
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([left, top, right, bottom], outline=(255,0,0), width=3)
    return overlay

def crop_with_white(img, left, top, right, bottom):
    w,h = img.size
    bg = Image.new("RGB", (w,h), "white")
    crop_area = img.crop((left, top, right, bottom))
    bg.paste(crop_area, (left, top))
    return bg.crop((left, top, right, bottom))

# ---------------- Main ----------------
if uploaded:
    original = Image.open(uploaded).convert("RGB")
    original = resize_large(original)
    w,h = original.size

    st.subheader("Original Photo with Crop Overlay")
    col1, col2 = st.columns(2)

    with col1:
        left = st.slider("Left", 0, w-1, 0)
        top = st.slider("Top", 0, h-1, 0)
        right = st.slider("Right", left+1, w, w)
        bottom = st.slider("Bottom", top+1, h, h)

    overlay_img = draw_overlay(original, left, top, right, bottom)
    st.image(overlay_img, use_column_width=True)

    # Apply flashy background and enhance
    if no_change:
        final = original
    else:
        bg_img = flashy_background(original, subject_type)
        enhanced = enhance(bg_img)
        # Crop manually
        cropped = crop_with_white(enhanced, left, top, right, bottom)
        # Resize to 630x810
        final = cropped.resize((630,810), Image.LANCZOS)
        final = add_border(final)

    with col2:
        st.subheader("Edited Passport Photo (630×810 px)")
        st.image(final, use_column_width=True)
        buf = io.BytesIO()
        final.save(buf, "JPEG", quality=95)
        st.download_button("Download Edited Photo", buf.getvalue(), file_name="edited_passport_photo.jpg", mime="image/jpeg")
