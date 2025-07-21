import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# Konfigurasi Streamlit
st.set_page_config(page_title="Pengolahan Citra", layout="wide")
st.title("🧠 Pengolahan Citra: Preprocessing, Segmentasi & Ekstraksi Ciri")
st.markdown("Upload gambar, sistem akan memproses dan menampilkan hasil segmentasi serta ciri-cirinya.")

# =======================
# DEFINISI FUNGSI
# =======================
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq = cv2.equalizeHist(blur)
    return eq

def segment_image(gray_image):
    _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def extract_features(image, mask):
    features = {}
    masked = cv2.bitwise_and(image, image, mask=mask)
    mean_val = cv2.mean(masked, mask=mask)
    features['Mean R'] = mean_val[0]
    features['Mean G'] = mean_val[1]
    features['Mean B'] = mean_val[2]

    features['Area'] = cv2.countNonZero(mask)
    x, y, w, h = cv2.boundingRect(mask)
    features['Aspect Ratio'] = round(w / h, 2) if h != 0 else 0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    features['Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    return features

# =======================
# UPLOAD & PROSES GAMBAR
# =======================
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)

    st.image(image, caption='Gambar Asli', use_container_width=True)

    # Panggil fungsi
    preprocessed = preprocess_image(img_np)
    segmented = segment_image(preprocessed)
    features = extract_features(img_np, segmented)

    col1, col2 = st.columns(2)
    with col1:
        st.image(preprocessed, caption='Preprocessing (Grayscale & Equalization)', use_container_width=True, clamp=True, channels="GRAY")
    with col2:
        st.image(segmented, caption='Segmentasi (Otsu Thresholding)', use_container_width=True, clamp=True, channels="GRAY")

    # Tampilkan hasil ekstraksi ciri
    st.subheader("📊 Ciri-ciri Gambar")
    for key, value in features.items():
        st.write(f"**{key}**: {value:.2f}" if isinstance(value, float) else f"**{key}**: {value}")
