import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px

# === KONFIGURASI APLIKASI ===
st.set_page_config(
    page_title="🍜 Nusantara Food Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# === KONFIGURASI PATH ===
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "budaya_nusantara_foods" / "dataset"
PREDICT_DIR = BASE_DIR / "budaya_nusantara_foods" / "predict_samples"

# File model berada di folder utama proyek
MODEL_PATH = BASE_DIR / "BestModel_CostumCNN_Pingouin.h5"
MODEL_NAME = "BestModel_CostumCNN_Pingouin.h5"

# === SIDEBAR INFO ===
with st.sidebar:
    st.title("⚙️ Info Model")
    st.write("**Model:** Custom CNN")
    st.write(f"**File:** {MODEL_NAME}")

    st.divider()
    st.subheader("📊 Info Dataset")
    st.write("**Kelas Makanan:**")
    st.write("- 🍛 Gudeg")
    st.write("- 🍲 Papeda")
    st.write("- 🐟 Pempek")
    st.write("- 🥘 Rendang")

    st.divider()
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write("Aplikasi ini menggunakan Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar makanan tradisional Indonesia.")

# === LOAD MODEL ===
@st.cache_resource
def load_model(model_path):
    if not model_path.exists():
        st.error(f"❌ File model tidak ditemukan di: {model_path}")
        st.info("💡 Pastikan file model `.h5` disimpan di folder proyek utama bersama file Streamlit.")
        st.stop()
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("💡 Pastikan model dibuat dengan TensorFlow versi kompatibel.")
        st.stop()

with st.spinner("⏳ Memuat model Custom CNN..."):
    model = load_model(MODEL_PATH)
    st.success("✅ Model Custom CNN berhasil dimuat!")

# === UTILITAS ===
def get_class_names():
    """Daftar kelas makanan"""
    return ["Gudeg", "Papeda", "Pempek", "Rendang"]

def preprocess_image(img: Image.Image, img_size=128):
    """Preprocessing gambar untuk prediksi"""
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def get_food_emoji(class_name):
    """Emoji berdasarkan kelas"""
    emoji_map = {
        "Gudeg": "🍛",
        "Papeda": "🍲",
        "Pempek": "🐟",
        "Rendang": "🥘"
    }
    return emoji_map.get(class_name, "🍜")

# === HEADER UTAMA ===
st.markdown('<p class="main-header">🍜 Nusantara Food Classifier 🍜</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Klasifikasi Makanan Tradisional Indonesia dengan Deep Learning</p>', unsafe_allow_html=True)

# === UPLOAD & PREDIKSI ===
st.header("📸 Upload dan Prediksi Gambar")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Gambar")
    uploaded_file = st.file_uploader(
        "Pilih gambar makanan:",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload gambar makanan yang ingin diprediksi"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diupload", use_container_width=True)

with col2:
    if uploaded_file is not None:
        st.subheader("🔍 Hasil Prediksi")
        with st.spinner("Menganalisis gambar..."):
            x = preprocess_image(img, 128)
            prob = model.predict(x, verbose=0)[0]
            class_names = get_class_names()
            pred_idx = int(np.argmax(prob))
            pred_class = class_names[pred_idx]
            confidence = float(np.max(prob)) * 100

            emoji = get_food_emoji(pred_class)
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #FF6B6B;">{emoji} {pred_class}</h2>
                <h3 style="text-align: center; color: #4ECDC4;">Confidence: {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

            if confidence > 80:
                st.success("🎯 Prediksi sangat yakin!")
            elif confidence > 60:
                st.info("👍 Prediksi cukup yakin")
            elif confidence > 40:
                st.warning("⚠️ Prediksi kurang yakin")
            else:
                st.error("❌ Prediksi sangat rendah, coba gambar lain")

# === GRAFIK PROBABILITAS ===
if uploaded_file is not None:
    st.divider()
    st.subheader("📊 Distribusi Probabilitas")

    class_names = get_class_names()
    prob_data = pd.DataFrame({
        'Kelas': class_names,
        'Probabilitas': prob * 100
    }).sort_values('Probabilitas', ascending=False)

    fig = px.bar(
        prob_data,
        x='Kelas',
        y='Probabilitas',
        color='Probabilitas',
        color_continuous_scale='RdYlGn',
        text='Probabilitas',
        title='Probabilitas untuk Setiap Kelas'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
