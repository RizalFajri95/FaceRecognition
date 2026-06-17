"""
Aplikasi Streamlit: Pengenalan Wajah dengan PCA (Eigenfaces)
==============================================================
Diadaptasi dari script Google Colab. Alur kerja:
  1. Upload foto "database" (wajah yang dikenal) + beri label/nama
  2. Latih model PCA (hitung mean face, eigenfaces, threshold)
  3. Upload foto "query" lalu lihat hasil pengenalannya
  4. (Opsional) lihat rekonstruksi wajah dengan berbagai jumlah komponen k

Jalankan dengan:
    streamlit run app.py
"""

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, aman untuk server
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# KONFIGURASI
# ============================================================

IMG_SIZE = (100, 100)
ACCENT = "#5340d4"
ACCENT_BAD = "#e85d30"
ACCENT_GOOD = "#3bb575"

st.set_page_config(
    page_title="Pengenalan Wajah - PCA Eigenfaces",
    page_icon="🧑‍💻",
    layout="wide",
)


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


FACE_CASCADE = load_face_cascade()


# ============================================================
# FUNGSI BANTU: GAMBAR & DETEKSI WAJAH
# ============================================================

def decode_uploaded_image(uploaded_file):
    """Ubah file hasil st.file_uploader jadi array gambar BGR (format OpenCV)."""
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def preprocess_face(img_bgr):
    """Deteksi wajah terbesar, crop, grayscale, equalize, resize, flatten.

    Mengembalikan (vektor_1D, gambar_grayscale_2D, wajah_terdeteksi: bool).
    Jika img_bgr gagal dibaca, mengembalikan (None, None, False).
    """
    if img_bgr is None:
        return None, None, False

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        face_crop = gray
        face_detected = False
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = gray[y:y + h, x:x + w]
        face_detected = True

    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_vector = face_resized.flatten().astype(np.float64)
    return face_vector, face_resized, face_detected


# ============================================================
# FUNGSI BANTU: PCA / EIGENFACES
# ============================================================

def train_pca(vectors):
    """Hitung mean face, eigenfaces (via SVD), dan proyeksi data database."""
    X = np.array(vectors)
    n_components = max(1, min(X.shape[0] - 1, 50))

    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    eigenvectors = Vt[:n_components].T
    X_projected = X_centered @ eigenvectors

    variance_ratio = (S[:n_components] ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(variance_ratio)

    return {
        "X": X,
        "mean_face": mean_face,
        "Vt": Vt,
        "S": S,
        "eigenvectors": eigenvectors,
        "X_projected": X_projected,
        "n_components": n_components,
        "variance_ratio": variance_ratio,
        "cumulative_var": cumulative_var,
    }


def calibrate_threshold(X_projected):
    """Cari jarak terdekat antar wajah berbeda di database, sarankan threshold
    di persentil ke-25. Mengembalikan (suggested_threshold, list_jarak) atau
    (None, []) kalau database hanya punya 1 foto."""
    n = len(X_projected)
    if n < 2:
        return None, []

    nearest_other_dists = []
    for i in range(n):
        dists_i = [
            np.linalg.norm(X_projected[i] - X_projected[j])
            for j in range(n) if j != i
        ]
        nearest_other_dists.append(min(dists_i))

    suggested = float(np.percentile(nearest_other_dists, 25))
    return suggested, nearest_other_dists


def recognize_face(query_vector, model, db_labels):
    """Proyeksikan vektor query ke ruang PCA, hitung jarak ke semua wajah
    database, urutkan dari terdekat. Mengembalikan list (jarak, label, idx)."""
    query_centered = query_vector - model["mean_face"]
    query_projected = query_centered @ model["eigenvectors"]

    distances = [
        (np.linalg.norm(query_projected - model["X_projected"][i]), db_labels[i], i)
        for i in range(len(model["X_projected"]))
    ]
    distances.sort(key=lambda x: x[0])
    return distances


# ============================================================
# FUNGSI BANTU: VISUALISASI (matplotlib -> st.pyplot)
# ============================================================

def fig_mean_and_eigenfaces(model):
    n_show = min(6, model["n_components"] + 1)
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    axes[0].imshow(model["mean_face"].reshape(IMG_SIZE), cmap="gray")
    axes[0].set_title("Mean Face")
    axes[0].axis("off")

    for i in range(1, n_show):
        ef = model["Vt"][i - 1].reshape(IMG_SIZE)
        rng = ef.max() - ef.min()
        ef = (ef - ef.min()) / rng if rng > 0 else np.zeros_like(ef)
        axes[i].imshow(ef, cmap="gray")
        axes[i].set_title(f"Eigenface {i}")
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def fig_variance(model):
    cumulative_var = model["cumulative_var"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100,
            marker="o", markersize=5, color=ACCENT)
    ax.axhline(90, color="coral", linestyle="--", label="90% variansi")
    ax.set_xlabel("Jumlah Komponen PCA")
    ax.set_ylabel("Variansi Kumulatif (%)")
    ax.set_title("Variansi yang Dijelaskan oleh Komponen PCA")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_threshold_hist(nearest_other_dists, suggested_threshold):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(nearest_other_dists, bins=min(10, len(nearest_other_dists)),
            color=ACCENT, alpha=0.7)
    ax.axvline(suggested_threshold, color="coral", linestyle="--",
               label=f"Saran threshold: {suggested_threshold:.0f}")
    ax.set_xlabel("Jarak terdekat ke wajah lain di database")
    ax.set_ylabel("Jumlah")
    ax.set_title("Distribusi jarak antar wajah (bantu menentukan threshold)")
    ax.legend()
    plt.tight_layout()
    return fig


def fig_recognition_result(query_img, best_img, best_label, distances, threshold):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(query_img, cmap="gray")
    axes[0].set_title("Foto Query")
    axes[0].axis("off")

    axes[1].imshow(best_img, cmap="gray")
    axes[1].set_title(f"Paling mirip: {best_label}")
    axes[1].axis("off")

    top_n = min(3, len(distances))
    top_labels = [d[1] for d in distances[:top_n]]
    top_dists = [d[0] for d in distances[:top_n]]
    colors = [ACCENT_GOOD if d < threshold else ACCENT_BAD for d in top_dists]
    axes[2].barh(top_labels[::-1], top_dists[::-1], color=colors[::-1])
    axes[2].axvline(threshold, color="gray", linestyle="--",
                     label=f"Threshold {threshold:.0f}")
    axes[2].set_title("Jarak ke database")
    axes[2].set_xlabel("Euclidean Distance")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    return fig


def fig_reconstruction(query_vector, model):
    mean_face = model["mean_face"]
    Vt = model["Vt"]
    n_components = model["n_components"]
    query_centered = query_vector - mean_face

    k_list = sorted(set(k for k in [1, 5, 10, 20, n_components] if k <= n_components))

    fig, axes = plt.subplots(1, len(k_list) + 1, figsize=(3 * (len(k_list) + 1), 3))

    axes[0].imshow(query_vector.reshape(IMG_SIZE), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, k in enumerate(k_list):
        proj_k = query_centered @ Vt[:k].T
        recon = proj_k @ Vt[:k] + mean_face
        recon = np.clip(recon, 0, 255)
        axes[i + 1].imshow(recon.reshape(IMG_SIZE), cmap="gray")
        axes[i + 1].set_title(f"k = {k}")
        axes[i + 1].axis("off")

    plt.suptitle("Rekonstruksi wajah dengan k komponen PCA", fontsize=13)
    plt.tight_layout()
    return fig


# ============================================================
# STATE AWAL
# ============================================================

if "model" not in st.session_state:
    st.session_state.model = None
if "db_labels" not in st.session_state:
    st.session_state.db_labels = []
if "db_images" not in st.session_state:
    st.session_state.db_images = []
if "suggested_threshold" not in st.session_state:
    st.session_state.suggested_threshold = None
if "nearest_other_dists" not in st.session_state:
    st.session_state.nearest_other_dists = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 5000.0
if "query_results" not in st.session_state:
    st.session_state.query_results = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Tentang aplikasi ini")
    st.markdown(
        "Aplikasi ini mengenali wajah memakai metode klasik **Eigenfaces "
        "(PCA)**: setiap foto wajah diproyeksikan ke ruang dimensi rendah "
        "hasil PCA, lalu foto query dicocokkan dengan jarak Euclidean "
        "terdekat ke foto-foto di database."
    )
    st.markdown(
        "**Langkah penggunaan:**\n\n"
        "1. Upload foto database & beri nama\n"
        "2. Klik **Latih Model PCA**\n"
        "3. Upload foto query\n"
        "4. Klik **Kenali Wajah**"
    )
    st.divider()
    if st.button("Reset semua data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.title("🧑‍💻 Pengenalan Wajah dengan PCA (Eigenfaces)")

# ============================================================
# LANGKAH 1: UPLOAD DATABASE
# ============================================================

st.header("1️⃣ Upload Foto Database (wajah yang dikenal)")

db_files = st.file_uploader(
    "Upload satu atau beberapa foto wajah untuk database",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="db_uploader",
)

db_label_inputs = {}
if db_files:
    st.caption("Atur nama/label untuk setiap foto (default = nama file):")
    cols_per_row = 4
    for row_start in range(0, len(db_files), cols_per_row):
        row_files = list(enumerate(db_files))[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_files))
        for col, (i, f) in zip(cols, row_files):
            with col:
                st.image(f.getvalue(), caption=f.name, width=130)
                default_label = os.path.splitext(f.name)[0]
                label = st.text_input(
                    "Nama", value=default_label, key=f"db_label_{i}_{f.name}"
                )
                db_label_inputs[i] = label

train_clicked = st.button("🚀 Latih Model PCA", type="primary", disabled=not db_files)

if train_clicked:
    vectors, labels, images = [], [], []
    skipped = []
    no_face_warnings = []

    for i, f in enumerate(db_files):
        img_bgr = decode_uploaded_image(f)
        vec, img_gray, face_found = preprocess_face(img_bgr)
        if vec is None:
            skipped.append(f.name)
            continue
        if not face_found:
            no_face_warnings.append(f.name)
        vectors.append(vec)
        labels.append(db_label_inputs.get(i, os.path.splitext(f.name)[0]))
        images.append(img_gray)

    if skipped:
        st.warning(f"Gagal membaca file: {', '.join(skipped)}")
    if no_face_warnings:
        st.info(
            f"Wajah tidak terdeteksi otomatis di: {', '.join(no_face_warnings)} "
            "— seluruh gambar dipakai sebagai gantinya."
        )

    if len(vectors) == 0:
        st.error(
            "Tidak ada foto valid di database. Pastikan file berekstensi "
            ".jpg/.jpeg/.png dan berhasil dibaca."
        )
    else:
        if len(vectors) < 2:
            st.warning(
                "Database hanya punya 1 foto. PCA butuh beberapa foto wajah "
                "berbeda untuk belajar variasi yang bermakna — hasil "
                "pengenalan dengan 1 foto kemungkinan besar tidak akurat."
            )

        model = train_pca(vectors)
        st.session_state.model = model
        st.session_state.db_labels = labels
        st.session_state.db_images = images

        suggested, nearest_dists = calibrate_threshold(model["X_projected"])
        st.session_state.suggested_threshold = suggested
        st.session_state.nearest_other_dists = nearest_dists
        st.session_state.threshold = round(suggested) if suggested is not None else 5000.0
        st.session_state.query_results = []  # reset hasil query lama

        st.success(
            f"Model selesai dilatih dari {len(vectors)} foto "
            f"({model['n_components']} komponen PCA)."
        )

# ============================================================
# LANGKAH 1B: HASIL ANALISIS PCA (kalau model sudah dilatih)
# ============================================================

model = st.session_state.model

if model is not None:
    st.subheader("Mean Face & Eigenfaces")
    fig = fig_mean_and_eigenfaces(model)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Variansi yang Dijelaskan PCA")
    fig = fig_variance(model)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"{model['n_components']} komponen menjelaskan "
        f"{model['cumulative_var'][-1] * 100:.1f}% variansi total."
    )

    st.subheader("Kalibrasi Threshold")
    if st.session_state.suggested_threshold is not None:
        nearest_dists = st.session_state.nearest_other_dists
        st.write(
            f"Rentang jarak antar wajah berbeda di database: "
            f"**{min(nearest_dists):.1f} – {max(nearest_dists):.1f}**"
        )
        st.write(
            f"Threshold yang disarankan (persentil 25): "
            f"**{st.session_state.suggested_threshold:.1f}**"
        )
        st.caption(
            "Catatan: ini estimasi awal dari database kamu sendiri. Tetap cek "
            "ulang dengan foto query asli (orang dikenal vs tidak dikenal), "
            "lalu sesuaikan slider di bawah kalau perlu."
        )
        fig = fig_threshold_hist(nearest_dists, st.session_state.suggested_threshold)
        st.pyplot(fig)
        plt.close(fig)

        slider_max = max(nearest_dists) * 2.5
    else:
        st.info("Database cuma 1 foto, tidak bisa dikalibrasi otomatis. Pakai nilai default.")
        slider_max = 10000.0

    st.session_state.threshold = st.slider(
        "Threshold pengenalan (jarak di bawah ini dianggap 'dikenali')",
        min_value=0.0,
        max_value=float(max(slider_max, st.session_state.threshold * 1.5, 100.0)),
        value=float(st.session_state.threshold),
        step=10.0,
    )

st.divider()

# ============================================================
# LANGKAH 2: UPLOAD & KENALI FOTO QUERY
# ============================================================

st.header("2️⃣ Upload Foto Query untuk Dikenali")

if model is None:
    st.info("Latih model PCA terlebih dahulu di Langkah 1 sebelum mengenali foto query.")
else:
    query_files = st.file_uploader(
        "Upload satu atau beberapa foto query",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="query_uploader",
    )

    recognize_clicked = st.button(
        "🔍 Kenali Wajah", type="primary", disabled=not query_files
    )

    if recognize_clicked:
        results = []
        for f in query_files:
            img_bgr = decode_uploaded_image(f)
            vec, img_gray, face_found = preprocess_face(img_bgr)
            if vec is None:
                st.warning(f"Gagal membaca: {f.name}")
                continue
            if not face_found:
                st.info(f"Wajah tidak terdeteksi di {f.name}, pakai seluruh gambar.")

            distances = recognize_face(vec, model, st.session_state.db_labels)
            best_dist, best_label, best_idx = distances[0]

            results.append({
                "file": f.name,
                "query_img": img_gray,
                "best_label": best_label,
                "best_idx": best_idx,
                "best_dist": best_dist,
                "distances": distances,
                "query_vec": vec,
            })

        st.session_state.query_results = results

    results = st.session_state.query_results

    if results:
        threshold = st.session_state.threshold

        for r in results:
            st.markdown(f"**{r['file']}**")
            best_img = st.session_state.db_images[r["best_idx"]]
            fig = fig_recognition_result(
                r["query_img"], best_img, r["best_label"], r["distances"], threshold
            )
            st.pyplot(fig)
            plt.close(fig)

            if r["best_dist"] < threshold:
                st.success(
                    f"WAJAH DIKENALI sebagai **{r['best_label']}** "
                    f"(jarak: {r['best_dist']:.1f}, threshold: {threshold:.0f})"
                )
            else:
                st.error(
                    f"WAJAH TIDAK DIKENALI — jarak terdekat {r['best_dist']:.1f} "
                    f"melebihi threshold {threshold:.0f}"
                )

        st.subheader("Ringkasan Hasil Pengenalan")
        summary_rows = [
            {
                "File": r["file"],
                "Prediksi": r["best_label"],
                "Jarak": round(r["best_dist"], 1),
                "Status": "DIKENALI" if r["best_dist"] < threshold else "TIDAK DIKENALI",
            }
            for r in results
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)

        st.divider()

        # ============================================================
        # LANGKAH 3 (opsional): REKONSTRUKSI WAJAH
        # ============================================================
        st.header("3️⃣ Rekonstruksi Wajah dengan Komponen PCA")
        st.caption(
            "Lihat bagaimana foto query disusun ulang memakai sebagian kecil "
            "komponen PCA. Semakin besar k, rekonstruksi semakin mendekati "
            "gambar asli."
        )
        file_names = [r["file"] for r in results]
        chosen_file = st.selectbox("Pilih foto query", file_names)
        chosen = next(r for r in results if r["file"] == chosen_file)

        fig = fig_reconstruction(chosen["query_vec"], model)
        st.pyplot(fig)
        plt.close(fig)
