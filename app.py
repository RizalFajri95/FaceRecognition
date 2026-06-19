"""
Aplikasi Pembanding Wajah (Foto Masa Kecil vs Foto Saat Ini)
=============================================================

Versi ini punya 2 mode:

1. MODE TRAINING
   - Upload dataset berupa file .zip terstruktur:
         dataset.zip
           |-- Andi/
           |     |-- foto1.jpg
           |     |-- foto2.jpg
           |-- Budi/
           |     |-- foto1.jpg
           |     ...
   - Semua wajah di dataset dipakai untuk melatih PCA (eigenfaces) sehingga
     model belajar variasi wajah manusia secara umum (pose, pencahayaan,
     bentuk wajah, dll) dari banyak orang & banyak foto.
   - Model (mean_face, eigenvectors, metadata) disimpan permanen ke disk
     (file .pkl) sehingga tidak perlu training ulang tiap kali aplikasi dibuka.

2. MODE BANDINGKAN (1 lawan 1)
   - Upload foto masa kecil & foto saat ini.
   - Kedua foto diproyeksikan ke ruang PCA yang SUDAH DILATIH dari dataset.
   - Jarak dihitung di ruang fitur PCA (bukan piksel mentah), sehingga jauh
     lebih tahan terhadap variasi pencahayaan/pose dan lebih representatif
     dibanding membandingkan piksel mentah secara langsung.

Catatan jujur soal akurasi:
   PCA/Eigenface tetap merupakan metode klasik (bukan deep learning).
   Melatihnya dengan dataset besar akan MENINGKATKAN akurasi dibanding
   versi 2-foto-saja, karena model jadi mengenal variasi wajah manusia
   secara umum. Namun untuk kasus sulit seperti membandingkan foto
   anak-anak vs dewasa (perubahan proporsi wajah akibat pertumbuhan),
   akurasi tetap akan jauh di bawah model deep learning modern
   (FaceNet/ArcFace) yang memang dilatih untuk age-invariant verification.
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import zipfile
import pickle
import tempfile
from datetime import datetime

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Pembanding Wajah: Masa Kecil vs Sekarang",
    page_icon="🧑‍🤝‍🧑",
    layout="wide",
)

IMG_SIZE = (100, 100)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")
MODEL_DIR = "model_store"
MODEL_PATH = os.path.join(MODEL_DIR, "pca_face_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

FACE_CASCADE = load_face_cascade()


# ----------------------------------------------------------------------
# Fungsi preprocessing wajah
# ----------------------------------------------------------------------
def preprocess_face_array(img_bgr):
    """Deteksi wajah, crop, equalize histogram, resize, flatten jadi vektor."""
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


def read_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def read_image_from_path(path):
    img = cv2.imread(path)
    return img


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf


# ----------------------------------------------------------------------
# Training PCA dari dataset ZIP
# ----------------------------------------------------------------------
def extract_zip(uploaded_zip, extract_to):
    zip_path = os.path.join(extract_to, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)


def find_dataset_root(extract_to):
    """
    Cari folder root yang berisi subfolder per-orang.
    Menangani kasus zip yang membungkus semuanya dalam 1 folder tambahan
    (umum terjadi saat zip dibuat dari folder di Windows/Mac).
    """
    entries = [
        e for e in os.listdir(extract_to)
        if not e.startswith("__MACOSX") and not e.startswith(".")
    ]
    subdirs = [e for e in entries if os.path.isdir(os.path.join(extract_to, e))]

    if len(subdirs) == 1:
        inner = os.path.join(extract_to, subdirs[0])
        inner_subdirs = [
            e for e in os.listdir(inner)
            if os.path.isdir(os.path.join(inner, e)) and not e.startswith(".")
        ]
        if inner_subdirs:
            return inner
    return extract_to


def load_dataset_faces(dataset_root, progress_callback=None):
    """
    Susuri dataset_root/<nama_orang>/*.jpg dan kembalikan:
    - vectors: list vektor wajah
    - labels: list nama orang (sesuai nama subfolder)
    - n_per_person: dict jumlah foto valid per orang
    - skipped: list file yang gagal dibaca / wajah tak terdeteksi
    - person_folders: list nama folder orang
    """
    person_folders = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    ])

    vectors, labels, skipped = [], [], []
    n_per_person = {}

    file_list = []
    for person in person_folders:
        person_path = os.path.join(dataset_root, person)
        files = sorted([
            f for f in os.listdir(person_path)
            if f.lower().endswith(VALID_EXT)
        ])
        for fname in files:
            file_list.append((person, os.path.join(person_path, fname)))

    total_files = len(file_list)

    for i, (person, fpath) in enumerate(file_list):
        img = read_image_from_path(fpath)
        if img is None:
            skipped.append(fpath + " (gagal dibaca)")
            continue
        vec, _, detected = preprocess_face_array(img)
        if not detected:
            skipped.append(fpath + " (wajah tidak terdeteksi, gambar penuh dipakai)")
        vectors.append(vec)
        labels.append(person)
        n_per_person[person] = n_per_person.get(person, 0) + 1

        if progress_callback:
            progress_callback((i + 1) / max(total_files, 1), person, fpath)

    return vectors, labels, n_per_person, skipped, person_folders


def train_pca_model(vectors, labels, n_components=None):
    X = np.array(vectors)
    n_samples = X.shape[0]

    if n_components is None:
        n_components = max(1, min(n_samples - 1, 100))
    else:
        n_components = max(1, min(n_components, n_samples - 1))

    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    eigenvectors = Vt[:n_components].T
    X_projected = X_centered @ eigenvectors

    variance_ratio = (S[:n_components] ** 2) / np.sum(S ** 2)
    cumulative_var = float(np.cumsum(variance_ratio)[-1]) * 100

    model = {
        "mean_face": mean_face,
        "eigenvectors": eigenvectors,
        "n_components": n_components,
        "X_projected": X_projected,
        "labels": labels,
        "img_size": IMG_SIZE,
        "n_samples": n_samples,
        "n_people": len(set(labels)),
        "cumulative_variance_pct": cumulative_var,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return model


def save_model(model, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def model_exists(path=MODEL_PATH):
    return os.path.exists(path)


# ----------------------------------------------------------------------
# Perbandingan 2 foto menggunakan model PCA terlatih
# ----------------------------------------------------------------------
def compare_with_trained_pca(model, vec_a, vec_b):
    mean_face = model["mean_face"]
    eigenvectors = model["eigenvectors"]

    proj_a = (vec_a - mean_face) @ eigenvectors
    proj_b = (vec_b - mean_face) @ eigenvectors

    euclidean_dist = float(np.linalg.norm(proj_a - proj_b))

    a_c = proj_a - proj_a.mean()
    b_c = proj_b - proj_b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    correlation = float(np.dot(a_c, b_c) / denom) if denom > 0 else 0.0

    return euclidean_dist, correlation, proj_a, proj_b


def estimate_threshold_from_model(model):
    """
    Estimasi threshold otomatis dari distribusi jarak antar wajah BERBEDA
    di dataset training (memakai label folder sebagai ground truth identitas).
    """
    X_projected = model["X_projected"]
    labels = model["labels"]
    n = len(labels)

    if n < 2:
        return 4000.0, [], []

    diff_person_dists = []
    same_person_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X_projected[i] - X_projected[j])
            if labels[i] == labels[j]:
                same_person_dists.append(d)
            else:
                diff_person_dists.append(d)

    if same_person_dists and diff_person_dists:
        threshold = (np.mean(same_person_dists) + np.mean(diff_person_dists)) / 2
    elif diff_person_dists:
        threshold = float(np.percentile(diff_person_dists, 10))
    else:
        threshold = 4000.0

    return float(threshold), same_person_dists, diff_person_dists


# ----------------------------------------------------------------------
# SIDEBAR: status model & navigasi
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Status Model")

current_model = load_model() if model_exists() else None

if current_model:
    st.sidebar.success("✅ Model PCA tersedia")
    st.sidebar.markdown(f"""
- **Dilatih dari**: {current_model['n_samples']} foto, {current_model['n_people']} orang
- **Komponen PCA**: {current_model['n_components']}
- **Variansi terjelaskan**: {current_model['cumulative_variance_pct']:.1f}%
- **Waktu training**: {current_model['trained_at']}
""")
    if st.sidebar.button("🗑️ Hapus Model (latih ulang dari awal)"):
        os.remove(MODEL_PATH)
        st.sidebar.warning("Model dihapus. Refresh halaman untuk melatih ulang.")
        st.rerun()
else:
    st.sidebar.warning("⚠️ Belum ada model terlatih. Silakan latih model di tab **Training** terlebih dahulu.")

mode = st.sidebar.radio("Pilih Mode", ["🏋️ Training Model", "🔍 Bandingkan 2 Foto"])


# ----------------------------------------------------------------------
# MODE: TRAINING
# ----------------------------------------------------------------------
if mode == "🏋️ Training Model":
    st.title("🏋️ Training Model PCA dari Dataset")
    st.markdown(
        """
Upload dataset wajah dalam bentuk **file .zip** dengan struktur folder seperti ini:

```
dataset.zip
 |-- Andi/
 |     |-- foto1.jpg
 |     |-- foto2.jpg
 |-- Budi/
 |     |-- foto1.jpg
 |     |-- foto2.jpg
 |-- Citra/
       |-- foto1.jpg
       ...
```

Semakin banyak **orang berbeda** dan **variasi foto per orang** (pose, pencahayaan, ekspresi),
semakin baik model PCA dapat belajar membedakan wajah secara umum.
"""
    )

    n_components_choice = st.slider(
        "Jumlah komponen PCA (semakin besar = menangkap lebih detail, tapi butuh lebih banyak data)",
        min_value=5, max_value=200, value=50, step=5
    )

    uploaded_zip = st.file_uploader("Upload dataset (.zip)", type=["zip"])

    if uploaded_zip is not None:
        if st.button("🚀 Mulai Training", type="primary"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with st.spinner("Mengekstrak dataset..."):
                    extract_zip(uploaded_zip, tmp_dir)
                    dataset_root = find_dataset_root(tmp_dir)

                person_folders_preview = sorted([
                    d for d in os.listdir(dataset_root)
                    if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
                ])

                if len(person_folders_preview) == 0:
                    st.error(
                        "Tidak ditemukan subfolder per-orang di dalam zip. "
                        "Pastikan struktur zip sesuai contoh di atas (setiap orang punya folder sendiri)."
                    )
                else:
                    st.info(
                        f"Ditemukan {len(person_folders_preview)} folder orang: "
                        f"{', '.join(person_folders_preview[:10])}"
                        f"{' ...' if len(person_folders_preview) > 10 else ''}"
                    )

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def progress_cb(frac, person, fpath):
                        progress_bar.progress(min(frac, 1.0))
                        status_text.text(f"Memproses: {person} -> {os.path.basename(fpath)}")

                    vectors, labels, n_per_person, skipped, person_folders = load_dataset_faces(
                        dataset_root, progress_callback=progress_cb
                    )
                    progress_bar.empty()
                    status_text.empty()

                    if len(vectors) < 2:
                        st.error(
                            f"Hanya ditemukan {len(vectors)} foto valid. "
                            "Minimal butuh beberapa foto dari beberapa orang untuk training PCA yang bermakna."
                        )
                    else:
                        with st.spinner("Melatih PCA..."):
                            model = train_pca_model(vectors, labels, n_components=n_components_choice)
                            save_model(model)

                        st.success(
                            f"✅ Training selesai! Model dilatih dari {model['n_samples']} foto, "
                            f"{model['n_people']} orang berbeda, {model['n_components']} komponen PCA "
                            f"(menjelaskan {model['cumulative_variance_pct']:.1f}% variansi)."
                        )

                        st.markdown("### 📋 Ringkasan Dataset")
                        summary_data = "\n".join(
                            f"- **{p}**: {n_per_person.get(p, 0)} foto" for p in person_folders
                        )
                        st.markdown(summary_data)

                        if skipped:
                            with st.expander(f"⚠️ {len(skipped)} catatan/peringatan saat membaca file"):
                                for s in skipped[:50]:
                                    st.text(s)
                                if len(skipped) > 50:
                                    st.text(f"... dan {len(skipped) - 50} lainnya")

                        st.markdown("### 👁️ Visualisasi Eigenfaces")
                        n_show = min(6, model["n_components"] + 1)
                        fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 2.5))
                        if n_show == 1:
                            axes = [axes]

                        axes[0].imshow(model["mean_face"].reshape(IMG_SIZE), cmap="gray")
                        axes[0].set_title("Mean Face")
                        axes[0].axis("off")

                        Vt_display = model["eigenvectors"].T
                        for i in range(1, n_show):
                            ef = Vt_display[i - 1].reshape(IMG_SIZE)
                            rng = ef.max() - ef.min()
                            ef = (ef - ef.min()) / rng if rng > 0 else np.zeros_like(ef)
                            axes[i].imshow(ef, cmap="gray")
                            axes[i].set_title(f"Eigenface {i}")
                            axes[i].axis("off")
                        plt.tight_layout()
                        st.image(fig_to_image(fig), use_container_width=True)

                        st.info(
                            "Model sudah disimpan secara permanen. Sekarang kamu bisa pindah ke tab "
                            "**'Bandingkan 2 Foto'** di sidebar kiri untuk menggunakannya."
                        )

    elif current_model:
        st.markdown("### Model saat ini sudah terlatih.")
        st.markdown(
            "Kamu bisa langsung lanjut ke mode **Bandingkan 2 Foto**, atau upload dataset baru "
            "di atas untuk melatih ulang model (akan menimpa model lama)."
        )


# ----------------------------------------------------------------------
# MODE: BANDINGKAN 2 FOTO
# ----------------------------------------------------------------------
elif mode == "🔍 Bandingkan 2 Foto":
    st.title("🔍 Bandingkan Wajah: Foto Masa Kecil vs Foto Saat Ini")

    if current_model is None:
        st.error(
            "Belum ada model PCA yang terlatih. Silakan ke tab **'Training Model'** "
            "di sidebar dan upload dataset (.zip) terlebih dahulu."
        )
        st.stop()

    st.success(
        f"Menggunakan model terlatih dari {current_model['n_samples']} foto, "
        f"{current_model['n_people']} orang ({current_model['n_components']} komponen PCA)."
    )

    with st.expander("ℹ️ Tentang metode & batasannya"):
        st.markdown(
            """
- Kedua foto diproyeksikan ke ruang fitur PCA yang sudah dilatih dari dataset besar,
  lalu dibandingkan jaraknya di ruang fitur tersebut (bukan piksel mentah).
- Ini lebih akurat dibanding membandingkan piksel mentah secara langsung, karena PCA
  menangkap pola variasi wajah manusia secara umum dari banyak data training.
- Threshold di bawah ini diestimasi otomatis dari data training (rata-rata antara
  jarak intra-orang dan jarak antar-orang berbeda), tapi tetap bisa kamu sesuaikan manual.
- Wajah anak-anak dan dewasa secara fisiologis berbeda proporsi, jadi skor kemiripan
  tetap merupakan **estimasi**, bukan jaminan verifikasi identitas 100% akurat.
"""
        )

    auto_threshold, same_dists, diff_dists = estimate_threshold_from_model(current_model)

    if same_dists and diff_dists:
        st.caption(
            f"📈 Dari data training: rata-rata jarak foto **orang sama** = {np.mean(same_dists):.0f}, "
            f"rata-rata jarak **orang berbeda** = {np.mean(diff_dists):.0f}. "
            f"Threshold otomatis = {auto_threshold:.0f}."
        )

    threshold = st.slider(
        "Ambang batas (threshold) jarak PCA untuk dianggap 'orang yang sama'",
        min_value=100, max_value=int(max(auto_threshold * 3, 2000)),
        value=int(auto_threshold), step=50,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Foto Masa Kecil")
        childhood_file = st.file_uploader("Upload foto masa kecil", type=["jpg", "jpeg", "png"], key="childhood")
    with col2:
        st.subheader("📷 Foto Saat Ini")
        current_file = st.file_uploader("Upload foto saat ini", type=["jpg", "jpeg", "png"], key="current")

    run_button = st.button("🔍 Bandingkan Wajah", type="primary", use_container_width=True)

    if run_button:
        if childhood_file is None or current_file is None:
            st.warning("Mohon upload kedua foto terlebih dahulu.")
        else:
            with st.spinner("Memproses gambar..."):
                img_child_bgr = read_image_from_upload(childhood_file)
                img_current_bgr = read_image_from_upload(current_file)

                vec_child, face_child, detected_child = preprocess_face_array(img_child_bgr)
                vec_current, face_current, detected_current = preprocess_face_array(img_current_bgr)

            if not detected_child:
                st.warning("⚠️ Wajah tidak terdeteksi pada foto masa kecil — seluruh gambar dipakai sebagai gantinya.")
            if not detected_current:
                st.warning("⚠️ Wajah tidak terdeteksi pada foto saat ini — seluruh gambar dipakai sebagai gantinya.")

            euclidean_dist, correlation, proj_child, proj_current = compare_with_trained_pca(
                current_model, vec_child, vec_current
            )
            max_ref_dist = max(auto_threshold * 2, euclidean_dist, 1.0)
            similarity_pct = max(0.0, min(100.0, 100.0 * (1 - euclidean_dist / max_ref_dist)))
            is_match = euclidean_dist < threshold

            st.markdown("---")
            st.header("📊 Hasil Perbandingan")

            img_col1, img_col2, img_col3 = st.columns([1, 1, 1.4])
            with img_col1:
                st.image(face_child, caption="Wajah Masa Kecil (diproses)", use_container_width=True, clamp=True)
            with img_col2:
                st.image(face_current, caption="Wajah Saat Ini (diproses)", use_container_width=True, clamp=True)
            with img_col3:
                diff = np.abs(face_child.astype(np.float64) - face_current.astype(np.float64)).astype(np.uint8)
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                ax.imshow(diff, cmap="inferno")
                ax.set_title("Peta Perbedaan (Difference Map)")
                ax.axis("off")
                st.image(fig_to_image(fig), use_container_width=True)

            st.markdown("### 🧮 Metrik Kemiripan (di ruang fitur PCA terlatih)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Jarak PCA (Euclidean)", f"{euclidean_dist:,.0f}")
            m2.metric("Korelasi (Pearson)", f"{correlation:.3f}")
            m3.metric("Skor Kemiripan", f"{similarity_pct:.1f}%")

            if is_match:
                st.success(
                    f"✅ **Kemungkinan ORANG YANG SAMA** — jarak ({euclidean_dist:,.0f}) "
                    f"di bawah threshold ({threshold:,})."
                )
            else:
                st.error(
                    f"❌ **Kemungkinan ORANG BERBEDA** — jarak ({euclidean_dist:,.0f}) "
                    f"melebihi threshold ({threshold:,})."
                )

            fig2, ax2 = plt.subplots(figsize=(6, 1.8))
            color = "#3bb575" if is_match else "#e85d30"
            ax2.barh(["Jarak PCA"], [euclidean_dist], color=color)
            ax2.axvline(threshold, color="gray", linestyle="--", label=f"Threshold ({threshold})")
            ax2.set_xlim(0, max(threshold * 1.5, euclidean_dist * 1.2))
            ax2.legend(loc="lower right", fontsize=8)
            ax2.set_xlabel("Jarak di ruang fitur PCA")
            st.image(fig_to_image(fig2), use_container_width=True)

            st.caption(
                "Catatan: skor ini estimasi berbasis model PCA yang dilatih dari dataset kamu, "
                "bukan verifikasi identitas yang dijamin 100% benar — terutama untuk foto lintas usia."
            )

st.markdown("---")
st.caption("Dibuat dengan Streamlit, OpenCV (Haar Cascade), dan PCA/Eigenface terlatih dari dataset.")
