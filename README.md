# Pembanding Wajah: Foto Masa Kecil vs Foto Saat Ini (dengan Training Dataset)

Aplikasi Streamlit untuk membandingkan kemiripan wajah dari dua foto (misalnya foto masa kecil dan foto saat ini), menggunakan model PCA/Eigenface yang **dilatih terlebih dahulu dari dataset wajah** agar hasilnya lebih akurat dibanding hanya membandingkan 2 foto secara langsung.

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

3. Buka browser ke alamat yang ditampilkan (biasanya `http://localhost:8501`).

## Cara Pakai

Aplikasi punya 2 mode (pilih di sidebar kiri):

### 1. 🏋️ Training Model
- Siapkan dataset wajah dalam **file .zip**, dengan struktur: setiap orang punya subfolder sendiri berisi beberapa foto.
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
- Upload zip tersebut, atur jumlah komponen PCA, lalu klik **"Mulai Training"**.
- Model (mean face, eigenvectors, dll) otomatis tersimpan permanen di `model_store/pca_face_model.pkl` — jadi tidak perlu training ulang setiap buka aplikasi.
- Semakin banyak orang & variasi foto per orang, semakin baik model mengenali pola wajah secara umum.

### 2. 🔍 Bandingkan 2 Foto
- Upload foto masa kecil dan foto saat ini.
- Kedua foto diproyeksikan ke ruang fitur PCA yang **sudah dilatih** dari dataset, lalu dibandingkan jaraknya di ruang fitur tersebut (bukan piksel mentah) — ini yang membuat hasil lebih akurat.
- Threshold kecocokan otomatis diestimasi dari data training, dan bisa disesuaikan manual lewat slider.
- Hasil mencakup: gambar wajah yang diproses, peta perbedaan visual, skor kemiripan, jarak PCA, korelasi, dan kesimpulan "kemungkinan sama" atau "kemungkinan berbeda".

## Catatan Akurasi

Melatih PCA dari dataset besar membuat model lebih mengenal variasi wajah manusia secara umum (pose, pencahayaan, bentuk wajah), sehingga **lebih akurat** dibanding membandingkan 2 foto piksel mentah secara langsung. Namun PCA/Eigenface tetap metode klasik (bukan deep learning) — untuk kasus sulit seperti foto anak-anak vs dewasa (proporsi wajah berubah akibat pertumbuhan), akurasinya tetap di bawah model deep learning modern seperti FaceNet/ArcFace yang memang dirancang untuk age-invariant face verification. Gunakan hasil aplikasi ini sebagai estimasi, bukan bukti identitas yang mutlak.
