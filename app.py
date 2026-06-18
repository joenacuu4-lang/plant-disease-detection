import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a5e20 0%, #2e7d32 50%, #43a047 100%);
    padding: 2.5rem 2rem; border-radius: 16px; color: white;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(30,94,32,0.3);
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }
.result-card {
    background: white; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 5px solid #2e7d32;
    margin-bottom: 1rem;
}
.result-card.danger  { border-left-color: #c62828; }
.result-card.warning { border-left-color: #f57f17; }
.result-card.success { border-left-color: #2e7d32; }
.metric-box {
    background: #f8fdf8; border-radius: 10px; padding: 1rem;
    text-align: center; border: 1px solid #c8e6c9;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1b5e20; }
.metric-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-healthy { background: #e8f5e9; color: #1b5e20; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.badge-disease { background: #ffebee; color: #b71c1c; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
.tip-box { background: #fff8e1; border: 1px solid #ffe082; border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1rem; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CLASS_LABELS = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy",
]

DISEASE_INFO = {
    "Apple___Apple_scab": {"severity":"warning","cause":"Jamur Venturia inaequalis","treatment":"Gunakan fungisida berbahan captan atau myclobutanil. Pangkas daun yang terinfeksi.","prevention":"Pastikan sirkulasi udara baik, hindari penyiraman berlebihan."},
    "Apple___Black_rot": {"severity":"danger","cause":"Jamur Botryosphaeria obtusa","treatment":"Hapus buah dan daun yang terinfeksi, gunakan fungisida copper-based.","prevention":"Sanitasi kebun, pangkas cabang mati secara rutin."},
    "Apple___Cedar_apple_rust": {"severity":"warning","cause":"Jamur Gymnosporangium juniperi-virginianae","treatment":"Aplikasi fungisida myclobutanil saat musim semi.","prevention":"Hindari menanam dekat pohon cedar/juniper."},
    "Cherry_(including_sour)___Powdery_mildew": {"severity":"warning","cause":"Jamur Podosphaera clandestina","treatment":"Semprot sulfur atau kalium bikarbonat.","prevention":"Jaga sirkulasi udara, hindari kelembaban berlebih."},
    "Corn_(maize)___Common_rust_": {"severity":"warning","cause":"Jamur Puccinia sorghi","treatment":"Fungisida triazole efektif jika diaplikasi dini.","prevention":"Gunakan varietas hibrida tahan karat, tanam lebih awal."},
    "Corn_(maize)___Northern_Leaf_Blight": {"severity":"warning","cause":"Jamur Exserohilum turcicum","treatment":"Fungisida propiconazole atau azoxystrobin.","prevention":"Rotasi tanaman, gunakan benih tahan penyakit."},
    "Grape___Black_rot": {"severity":"danger","cause":"Jamur Guignardia bidwellii","treatment":"Fungisida mancozeb atau captan sejak tunas muncul.","prevention":"Pangkas tanaman, buang buah yang terinfeksi."},
    "Orange___Haunglongbing_(Citrus_greening)": {"severity":"danger","cause":"Bakteri Candidatus Liberibacter","treatment":"Tidak ada pengobatan, cabut pohon yang terinfeksi.","prevention":"Kendalikan serangga vektor, gunakan bibit bersertifikat."},
    "Peach___Bacterial_spot": {"severity":"warning","cause":"Bakteri Xanthomonas arboricola","treatment":"Copper hydroxide saat daun muncul.","prevention":"Pilih varietas tahan, hindari luka mekanis."},
    "Pepper,_bell___Bacterial_spot": {"severity":"warning","cause":"Bakteri Xanthomonas euvesicatoria","treatment":"Copper-based bactericide, hindari irigasi overhead.","prevention":"Rotasi tanaman, gunakan benih bebas penyakit."},
    "Potato___Early_blight": {"severity":"warning","cause":"Jamur Alternaria solani","treatment":"Fungisida mancozeb atau chlorothalonil.","prevention":"Rotasi tanaman 2-3 tahun, mulsa tanah."},
    "Potato___Late_blight": {"severity":"danger","cause":"Oomycete Phytophthora infestans","treatment":"Semprot fungisida sistemik segera, cabut tanaman terinfeksi berat.","prevention":"Gunakan benih bersertifikat, rotasi tanaman, drainase baik."},
    "Squash___Powdery_mildew": {"severity":"warning","cause":"Jamur Erysiphe cichoracearum","treatment":"Semprot neem oil atau kalium bikarbonat.","prevention":"Tanam dengan jarak cukup, hindari pemupukan nitrogen berlebih."},
    "Strawberry___Leaf_scorch": {"severity":"warning","cause":"Jamur Diplocarpon earlianum","treatment":"Fungisida captan atau myclobutanil.","prevention":"Hindari penyiraman dari atas, sanitasi sisa tanaman."},
    "Tomato___Bacterial_spot": {"severity":"warning","cause":"Bakteri Xanthomonas vesicatoria","treatment":"Copper-based bactericide, hindari irigasi overhead.","prevention":"Rotasi tanaman, gunakan benih bebas patogen."},
    "Tomato___Early_blight": {"severity":"warning","cause":"Jamur Alternaria solani","treatment":"Aplikasi fungisida mancozeb atau chlorothalonil.","prevention":"Rotasi tanaman 2-3 tahun, mulsa tanah, hindari penyiraman dari atas."},
    "Tomato___Late_blight": {"severity":"danger","cause":"Oomycete Phytophthora infestans","treatment":"Gunakan fungisida chlorothalonil atau copper-based segera.","prevention":"Rotasi tanaman, hindari kelembapan berlebih, gunakan varietas tahan."},
    "Tomato___Leaf_Mold": {"severity":"warning","cause":"Jamur Passalora fulva","treatment":"Fungisida chlorothalonil, tingkatkan ventilasi.","prevention":"Kurangi kelembaban, pastikan ventilasi greenhouse baik."},
    "Tomato___Septoria_leaf_spot": {"severity":"warning","cause":"Jamur Septoria lycopersici","treatment":"Fungisida copper atau chlorothalonil saat gejala muncul.","prevention":"Rotasi tanaman, mulsa, hindari penyiraman dari atas."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"severity":"warning","cause":"Tungau Tetranychus urticae","treatment":"Akarisida atau sabun insektisida, semprotkan air kuat.","prevention":"Jaga kelembaban, kendalikan semut, gunakan predator alami."},
    "Tomato___Target_Spot": {"severity":"warning","cause":"Jamur Corynespora cassiicola","treatment":"Fungisida azoxystrobin atau difenoconazole.","prevention":"Rotasi tanaman, sanitasi sisa panen."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"severity":"danger","cause":"Virus TYLCV (ditularkan kutu kebul)","treatment":"Tidak ada pengobatan virus, cabut tanaman terinfeksi.","prevention":"Kendalikan Bemisia tabaci (kutu kebul), gunakan mulsa reflektif."},
    "Tomato___Tomato_mosaic_virus": {"severity":"danger","cause":"Virus ToMV","treatment":"Tidak ada pengobatan, musnahkan tanaman terinfeksi.","prevention":"Cuci tangan sebelum menangani tanaman, gunakan benih bersertifikat."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"severity":"warning","cause":"Jamur Cercospora zeae-maydis","treatment":"Fungisida strobilurin atau triazole.","prevention":"Rotasi tanaman, gunakan varietas tahan."},
    "Grape___Esca_(Black_Measles)": {"severity":"danger","cause":"Kompleks jamur Phaeomoniella dan Phaeoacremonium","treatment":"Tidak ada pengobatan efektif, pangkas bagian yang terinfeksi.","prevention":"Hindari luka saat pemangkasan, gunakan pasta pelindung luka."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"severity":"warning","cause":"Jamur Isariopsis clavispora","treatment":"Fungisida tembaga atau mancozeb.","prevention":"Sanitasi kebun, pangkas untuk meningkatkan sirkulasi udara."},
    "Peach___Bacterial_spot": {"severity":"warning","cause":"Bakteri Xanthomonas arboricola pv. pruni","treatment":"Copper hydroxide saat tunas pecah.","prevention":"Pilih varietas tahan, hindari cedera mekanis."},
}

HEALTHY_INFO = {
    "severity":"success",
    "cause":"-",
    "treatment":"Tidak diperlukan penanganan khusus.",
    "prevention":"Pertahankan kondisi tanam yang baik: penyiraman teratur, nutrisi cukup, dan cahaya matahari optimal."
}

MODEL_PATH = "models/plant_disease_model.h5"

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def format_label(label):
    parts = label.split("___")
    plant = parts[0].replace("_", " ").replace(",", "")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in condition.lower()
    return plant, condition, is_healthy

def get_top_predictions(predictions, top_k=3):
    indices = np.argsort(predictions[0])[::-1][:top_k]
    return [(CLASS_LABELS[i], float(predictions[0][i])) for i in indices]

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🌿 Plant Disease Detector")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- 🏗️ **Arsitektur:** EfficientNetB4 + CBAM\n- 📊 **Dataset:** PlantVillage\n- 🌱 **Kelas:** 38 kondisi\n- 🎯 **Target Akurasi:** >95%")
    st.markdown("---")
    st.markdown("**Tanaman Didukung**")
    for p in ["🍎 Apple","🫐 Blueberry","🍒 Cherry","🌽 Corn","🍇 Grape","🍊 Orange","🍑 Peach","🌶️ Pepper","🥔 Potato","🫐 Raspberry","🫘 Soybean","🎃 Squash","🍓 Strawberry","🍅 Tomato"]:
        st.markdown(f"  {p}")
    st.markdown("---")
    st.caption("🎓 UAS Artificial Intelligence")

# ── HEADER ──
st.markdown('<div class="main-header"><h1>🌿 Plant Disease Detection System</h1><p>Deteksi penyakit tanaman menggunakan EfficientNetB4 + CBAM Attention Mechanism</p></div>', unsafe_allow_html=True)

# ── METRICS ──
c1,c2,c3,c4 = st.columns(4)
for col, val, lbl in zip([c1,c2,c3,c4],["38","54K+","14","95%+"],["Kelas Penyakit","Gambar Training","Jenis Tanaman","Target Akurasi"]):
    col.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_up, col_res = st.columns([1,1], gap="large")

with col_up:
    st.markdown("### 📸 Upload Foto Daun")
    uploaded_file = st.file_uploader("Pilih gambar daun tanaman", type=["jpg","jpeg","png","webp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        st.markdown(f'<div style="background:#f5f5f5;border-radius:8px;padding:0.8rem;font-size:0.85rem;color:#555;">📄 <b>{uploaded_file.name}</b> | 📐 {image.size[0]}×{image.size[1]} px | 💾 {uploaded_file.size//1024} KB</div>', unsafe_allow_html=True)
        detect_btn = st.button("🔍 Deteksi Penyakit", type="primary", use_container_width=True)
    else:
        st.markdown('<div style="background:#f1f8e9;border:2px dashed #a5d6a7;border-radius:12px;padding:2rem;text-align:center;"><div style="font-size:3rem;">🌱</div><p style="color:#558b2f;font-weight:500;margin:0.5rem 0;">Drag & drop foto daun di sini</p><p style="color:#999;font-size:0.85rem;margin:0;">atau klik untuk browse file</p></div>', unsafe_allow_html=True)
        detect_btn = False

with col_res:
    st.markdown("### 📊 Hasil Deteksi")
    if uploaded_file and detect_btn:
        with st.spinner("🔬 Menganalisis gambar..."):
            time.sleep(0.5)
            model = load_model()
            if model is None:
                st.warning("⚠️ **Demo Mode** — Model belum tersedia. Menampilkan simulasi hasil.")
                top_label = "Tomato___Late_blight"
                confidence = 0.872
                top_preds = [("Tomato___Late_blight",0.872),("Tomato___Early_blight",0.091),("Tomato___healthy",0.021)]
            else:
                img_array = preprocess_image(image)
                predictions = model.predict(img_array, verbose=0)
                top_preds = get_top_predictions(predictions, top_k=3)
                top_label, confidence = top_preds[0]

        plant, condition, is_healthy = format_label(top_label)
        info = HEALTHY_INFO if is_healthy else DISEASE_INFO.get(top_label, {"severity":"warning","cause":"Patogen tanaman","treatment":"Konsultasikan dengan ahli pertanian.","prevention":"Jaga kebersihan lahan dan rotasi tanaman."})
        severity = info["severity"]
        badge = '<span class="badge-healthy">✅ Sehat</span>' if is_healthy else '<span class="badge-disease">⚠️ Terdeteksi Penyakit</span>'

        st.markdown(f'<div class="result-card {severity}">{badge}<h3 style="margin:0.8rem 0 0.3rem;color:#1a1a1a;">🌱 {plant}</h3><p style="font-size:1.1rem;font-weight:600;color:#333;margin:0;">{condition}</p><p style="font-size:1.8rem;font-weight:700;color:#1b5e20;margin:0.5rem 0 0;">{confidence*100:.1f}% <span style="font-size:0.9rem;font-weight:400;color:#666;">confidence</span></p></div>', unsafe_allow_html=True)

        st.markdown("**Top-3 Prediksi:**")
        for lbl, conf in top_preds:
            _, cond, hlthy = format_label(lbl)
            color = "#2e7d32" if hlthy else "#c62828"
            st.markdown(f"<small style='color:#666;'>{cond}</small>", unsafe_allow_html=True)
            st.progress(conf)
            st.markdown(f"<small style='color:{color};font-weight:600;'>{conf*100:.1f}%</small>", unsafe_allow_html=True)

        if not is_healthy:
            st.markdown("---")
            i1, i2 = st.columns(2)
            with i1: st.markdown(f"🦠 **Penyebab:**  \n{info['cause']}")
            with i2: st.markdown(f"🛡️ **Pencegahan:**  \n{info['prevention']}")
            st.markdown(f'<div class="tip-box"><b>💊 Rekomendasi Penanganan:</b><br>{info["treatment"]}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Tanaman dalam kondisi sehat!")
            st.markdown(f'<div class="tip-box"><b>💡 Tips Perawatan:</b><br>{info["prevention"]}</div>', unsafe_allow_html=True)

    elif not uploaded_file:
        st.markdown('<div style="background:#f8f9fa;border-radius:12px;padding:3rem;text-align:center;color:#aaa;"><div style="font-size:3rem;">🔬</div><p style="margin:0.5rem 0 0;font-size:0.95rem;">Upload gambar daun untuk melihat hasil deteksi</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### ⚙️ Cara Kerja Model")
hw1,hw2,hw3,hw4 = st.columns(4)
for col, num, title, desc in zip([hw1,hw2,hw3,hw4],
    ["1️⃣","2️⃣","3️⃣","4️⃣"],
    ["Input Gambar","Preprocessing","EfficientNetB4+CBAM","Klasifikasi"],
    ["Upload foto daun JPG/PNG","Resize 224×224px, normalisasi","Ekstraksi fitur + attention","Output 38 kelas penyakit"]):
    col.markdown(f'<div style="background:white;border-radius:10px;padding:1.2rem;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.06);border-top:3px solid #2e7d32;"><div style="font-size:1.5rem;">{num}</div><b style="color:#1b5e20;">{title}</b><p style="font-size:0.8rem;color:#666;margin:0.5rem 0 0;">{desc}</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🎓 Plant Disease Detection | UAS Artificial Intelligence | EfficientNetB4 + CBAM Attention Mechanism")
