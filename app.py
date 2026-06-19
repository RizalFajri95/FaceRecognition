"""
Aplikasi Perbandingan Kemiripan Wajah (Foto Masa Kecil vs Foto Saat Ini)
Menggunakan metode Eigenface (PCA) di atas OpenCV Haar Cascade untuk deteksi wajah.

Catatan penting:
- PCA/Eigenface klasik biasanya dilatih di atas BANYAK wajah untuk menangkap
  variasi yang bermakna. Di sini kita hanya punya 2 foto (anak vs dewasa),
  jadi PCA tidak bisa membangun basis komponen yang kaya. Sebagai gantinya,
  aplikasi ini menghitung kemiripan dengan membandingkan vektor piksel wajah
  yang sudah dinormalisasi (jarak Euclidean & korelasi), DAN tetap
  menampilkan visualisasi gaya "eigenface" (mean face, dsb) sebagai bonus
  edukatif -- bukan sebagai metode akurasi utama.
- Untuk hasil pengenalan wajah yang benar-benar akurat (apalagi membandingkan
  foto anak-anak vs dewasa yang sama orangnya), dibutuhkan model deep learning
  khusus (mis. FaceNet/ArcFace) yang dilatih untuk invariansi usia. Skor di
  sini adalah perkiraan kasar berbasis kemiripan piksel, bukan verifikasi
  identitas yang terjamin benar.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Pembanding Wajah: Masa Kecil vs Sekarang",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)

IMG_SIZE = (100, 100)

# ----------------------------------------------------------------------
# Load face cascade (cached supaya tidak load ulang tiap interaksi)
# ----------------------------------------------------------------------
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

FACE_CASCADE = load_face_cascade()


# ----------------------------------------------------------------------
# Fungsi-fungsi inti (diadaptasi dari skrip Colab asli)
# ----------------------------------------------------------------------
def read_image_from_upload(uploaded_file):
    """Baca file upload Streamlit jadi array BGR (format OpenCV)."""
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr


def preprocess_face(img_bgr, label="gambar"):
    """Deteksi wajah, crop, equalize histogram, resize, lalu flatten jadi vektor."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    face_detected = True
    if len(faces) == 0:
        face_detected = False
        face_crop = gray
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y + h, x:x + w]

    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_vector = face_resized.flatten().astype(np.float64)

    return face_vector, face_resized, face_detected


def compute_similarity(vec_a, vec_b):
    """
    Hitung beberapa metrik kemiripan antara dua vektor wajah:
    - Euclidean distance (semakin kecil = semakin mirip)
    - Korelasi (cosine-like, via Pearson correlation) -> -1..1, makin
      mendekati 1 makin mirip
    - Skor kemiripan dalam persen (0-100%), hasil normalisasi jarak.
    """
    euclidean_dist = float(np.linalg.norm(vec_a - vec_b))

    a_centered = vec_a - vec_a.mean()
    b_centered = vec_b - vec_b.mean()
    denom = (np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    correlation = float(np.dot(a_centered, b_centered) / denom) if denom > 0 else 0.0

    # Normalisasi jarak euclidean menjadi skor persen.
    # Untuk gambar grayscale 100x100 (rentang piksel 0-255), jarak maksimum
    # teoritis sangat besar, tapi dalam praktiknya wajah-wajah berbeda biasanya
    # berjarak di kisaran 3000-9000. Kita pakai kurva normalisasi yang masuk
    # akal secara empiris, bukan jarak maksimum teoritis (yang akan membuat
    # semua skor terlihat sangat tinggi / tidak informatif).
    max_expected_dist = 8000.0
    similarity_pct = max(0.0, 100.0 * (1 - euclidean_dist / max_expected_dist))
    similarity_pct = min(similarity_pct, 100.0)

    return euclidean_dist, correlation, similarity_pct


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🧑‍🤝‍🧑 Pembanding Wajah: Foto Masa Kecil vs Foto Sekarang")
st.markdown(
    """
Unggah **foto masa kecil** dan **foto saat ini**, lalu aplikasi akan:
1. Mendeteksi wajah pada kedua foto
2. Mengekstrak fitur wajah (eigenface-style)
3. Menghitung skor kemiripan antara keduanya
"""
)

with st.expander("ℹ️ Tentang metode & batasannya (penting dibaca)"):
    st.markdown(
        """
- Metode ini berbasis **PCA/Eigenface** sederhana di atas piksel wajah yang sudah
  dinormalisasi (grayscale, equalize histogram, resize ke 100x100).
- Karena hanya membandingkan **2 foto**, PCA tidak dilatih atas banyak data, jadi
  skor kemiripan dihitung langsung dari jarak vektor piksel wajah yang sudah
  diproses — bukan dari proyeksi PCA penuh seperti pada skrip aslinya yang
  berisi banyak foto di database.
- Wajah anak-anak dan wajah dewasa secara fisiologis **berbeda** (proporsi wajah
  berubah seiring pertumbuhan), sehingga skor kemiripan piksel **tidak akan
  setinggi** membandingkan dua foto dewasa yang sama. Gunakan skor ini sebagai
  estimasi kasar, bukan bukti identitas yang pasti.
- Untuk akurasi tinggi pada kasus *age-invariant face verification*, idealnya
  dipakai model deep learning yang memang dilatih untuk itu.
"""
    )

col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Foto Masa Kecil")
    childhood_file = st.file_uploader(
        "Upload foto masa kecil", type=["jpg", "jpeg", "png"], key="childhood"
    )
with col2:
    st.subheader("📷 Foto Saat Ini")
    current_file = st.file_uploader(
        "Upload foto saat ini", type=["jpg", "jpeg", "png"], key="current"
    )

threshold = st.slider(
    "Ambang batas (threshold) jarak untuk dianggap 'kemungkinan orang yang sama'",
    min_value=1000, max_value=10000, value=6000, step=100,
    help="Semakin kecil jarak Euclidean dibanding threshold ini, semakin mirip kedua wajah."
)

run_button = st.button("🔍 Bandingkan Wajah", type="primary", use_container_width=True)

if run_button:
    if childhood_file is None or current_file is None:
        st.warning("Mohon upload kedua foto terlebih dahulu (masa kecil dan saat ini).")
    else:
        with st.spinner("Memproses gambar..."):
            img_child_bgr = read_image_from_upload(childhood_file)
            img_current_bgr = read_image_from_upload(current_file)

            vec_child, face_child, detected_child = preprocess_face(img_child_bgr, "masa kecil")
            vec_current, face_current, detected_current = preprocess_face(img_current_bgr, "saat ini")

        # Peringatan jika wajah tidak terdeteksi
        if not detected_child:
            st.warning("⚠️ Wajah tidak terdeteksi pada foto masa kecil — seluruh gambar dipakai sebagai gantinya. Hasil mungkin kurang akurat.")
        if not detected_current:
            st.warning("⚠️ Wajah tidak terdeteksi pada foto saat ini — seluruh gambar dipakai sebagai gantinya. Hasil mungkin kurang akurat.")

        # Hitung kemiripan
        euclidean_dist, correlation, similarity_pct = compute_similarity(vec_child, vec_current)
        is_match = euclidean_dist < threshold

        st.markdown("---")
        st.header("📊 Hasil Perbandingan")

        # Tampilkan kedua wajah yang sudah diproses
        img_col1, img_col2, img_col3 = st.columns([1, 1, 1.4])
        with img_col1:
            st.image(face_child, caption="Wajah Masa Kecil (diproses)", use_container_width=True, clamp=True)
        with img_col2:
            st.image(face_current, caption="Wajah Saat Ini (diproses)", use_container_width=True, clamp=True)
        with img_col3:
            # Visualisasi selisih absolut antar wajah
            diff = np.abs(face_child.astype(np.float64) - face_current.astype(np.float64)).astype(np.uint8)
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            im = ax.imshow(diff, cmap="inferno")
            ax.set_title("Peta Perbedaan (Difference Map)")
            ax.axis("off")
            st.image(fig_to_image(fig), use_container_width=True)

        st.markdown("### 🧮 Metrik Kemiripan")
        m1, m2, m3 = st.columns(3)
        m1.metric("Jarak Euclidean", f"{euclidean_dist:,.0f}")
        m2.metric("Korelasi (Pearson)", f"{correlation:.3f}")
        m3.metric("Skor Kemiripan", f"{similarity_pct:.1f}%")

        if is_match:
            st.success(
                f"✅ **Kemungkinan ORANG YANG SAMA** — jarak ({euclidean_dist:,.0f}) "
                f"berada di bawah threshold ({threshold:,})."
            )
        else:
            st.error(
                f"❌ **Kemungkinan ORANG BERBEDA** — jarak ({euclidean_dist:,.0f}) "
                f"melebihi threshold ({threshold:,})."
            )

        # Bar chart visual jarak vs threshold
        fig2, ax2 = plt.subplots(figsize=(6, 1.8))
        color = "#3bb575" if is_match else "#e85d30"
        ax2.barh(["Jarak"], [euclidean_dist], color=color)
        ax2.axvline(threshold, color="gray", linestyle="--", label=f"Threshold ({threshold})")
        ax2.set_xlim(0, max(threshold * 1.5, euclidean_dist * 1.2))
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_xlabel("Euclidean Distance")
        st.image(fig_to_image(fig2), use_container_width=True)

        st.caption(
            "Catatan: skor kemiripan adalah estimasi berbasis piksel wajah yang "
            "dinormalisasi, bukan verifikasi identitas yang pasti benar — "
            "terutama untuk perbandingan lintas usia (anak-anak vs dewasa)."
        )

st.markdown("---")
st.caption("Dibuat dengan Streamlit, OpenCV (Haar Cascade), dan prinsip Eigenface/PCA.")
