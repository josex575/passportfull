import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import cv2
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

def detect_face(img):
    """Detect face using OpenCV"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces)==0:
        return None
    return max(faces, key=lambda r: r[2]*r[3])

def change_background(img, color=(255, 0, 255)):
    """Change background to flashy color using GrabCut"""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    rect = (10,10,w-10,h-10)
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask_f = cv2.GaussianBlur(mask2.astype(np.float32),(7,7),0)[...,np.newaxis]
        flashy_bg = np.ones_like(arr)*np.array(color, dtype=np.uint8)
        comp = (arr*mask_f + flashy_bg*(1-mask_f)).astype(np.uint8)
        return Image.fromarray(comp)
    except:
        st.warning("Background replacement failed. Returning original")
        return img

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
        face = detect_face(original)
        if face is None:
            st.warning("Face not detected. Editing may be inaccurate.")
            final_img = original
        else:
            # Determine flashy color based on subject type
            color_map = {"Man": (0,255,255), "Woman": (255,105,180), "Baby": (255,255,0)}
            color = color_map.get(subject_type, (255,0,255))

            edited = change_background(original, color=color)
            final_img = enhance(edited)

    st.subheader("Edited Photo Preview")
    st.image(final_img, use_column_width=True)

    buf = io.BytesIO()
    final_img.save(buf, "JPEG", quality=95)
    st.download_button("Download Edited Photo", buf.getvalue(), file_name="edited_passport_photo.jpg", mime="image/jpeg")
