# Pembanding Wajah: Foto Masa Kecil vs Foto Saat Ini

Aplikasi Streamlit untuk membandingkan kemiripan wajah dari dua foto (misalnya foto masa kecil dan foto saat ini) menggunakan pendekatan Eigenface/PCA di atas deteksi wajah Haar Cascade (OpenCV).

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

1. Upload foto masa kecil di kolom kiri.
2. Upload foto saat ini di kolom kanan.
3. (Opsional) Sesuaikan slider threshold jarak.
4. Klik tombol **"Bandingkan Wajah"**.
5. Lihat hasil: wajah yang sudah diproses, peta perbedaan, skor kemiripan, dan kesimpulan kemungkinan orang yang sama atau berbeda.

## Catatan

Pendekatan ini cocok untuk demo/edukasi PCA & computer vision, namun **bukan** sistem verifikasi identitas yang presisi tinggi — terutama untuk perbandingan lintas usia (anak-anak vs dewasa), karena proporsi wajah berubah seiring pertumbuhan. Untuk kebutuhan produksi/akurasi tinggi, sebaiknya gunakan model deep learning khusus face recognition (misalnya berbasis FaceNet/ArcFace).
