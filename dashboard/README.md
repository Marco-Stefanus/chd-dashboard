# Dashboard Prediksi CHD: XGBoost, EBM, TabNet

Aplikasi Streamlit untuk menyajikan hasil skripsi: perbandingan tiga model dalam memprediksi
Coronary Heart Disease (TenYearCHD) pada data Framingham.

## Isi aplikasi

| Halaman | Isi |
|---|---|
| Ringkasan | Temuan utama dan tabel hasil test set |
| Perbandingan model | Metrik, kasus terlewat vs alarm palsu, confusion matrix, ROC, cek overfitting |
| Detail per model | Hyperparameter terbaik dan metrik tiap model |
| Interpretabilitas | SHAP, LIME, dan importance EBM |
| Prediksi pasien | Input data pasien, prediksi 3 model, majority voting |
| Metodologi dan catatan | Alur pra-pemrosesan dan keterbatasan |

Semua angka di dashboard diambil dari output notebook (`results_data.py`).
Halaman **Prediksi pasien** memakai model terlatih, jadi langkah 1 di bawah wajib dilakukan.

## 1. Simpan model dari notebook (sekali saja)

Buka notebook skripsi Anda, lalu jalankan isi file `export_models_cell.py` sebagai sel terakhir
(setelah `best_xgb`, `best_ebm`, `tabnet_model`, dan `scaler` ada di memori).
Hasilnya empat file di folder `models/`:

```
models/xgb_model.json
models/ebm_model.joblib
models/scaler.joblib
models/tabnet_model.zip
```

Jika sebagian file belum ada, aplikasi tetap jalan: model yang tersedia dipakai, sisanya ditandai
"belum termuat". Tanpa satu pun file model, halaman prediksi menampilkan kasus contoh dari notebook.

## 2. Jalankan di komputer sendiri

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501.

> **Samakan versi library.** Model EBM (joblib) dan TabNet sensitif terhadap versi.
> Di environment notebook jalankan `pip freeze`, lalu salin versi `xgboost`, `interpret`,
> `pytorch-tabnet`, `torch`, dan `scikit-learn` ke `requirements.txt` (mis. `xgboost==2.1.1`).
> Pakai versi Python yang sama juga.

## 3. Deploy

### Streamlit Community Cloud (disarankan, gratis)

1. Buat repository GitHub, unggah seluruh isi folder ini (termasuk `models/` dan `assets/`).
2. Buka https://share.streamlit.io, login dengan GitHub, pilih **Create app**.
3. Pilih repository, branch `main`, dan *Main file path* `app.py`.
4. Di **Advanced settings** pilih versi Python yang sama dengan notebook, lalu **Deploy**.

Anda mendapat tautan publik `https://nama-app.streamlit.app` yang bisa dicantumkan di skripsi.

### Hugging Face Spaces atau Render (alternatif)

`Dockerfile` sudah disediakan. Di Hugging Face buat Space dengan SDK **Docker**,
atau di Render buat *Web Service* dari repository yang sama.

### Vercel

**Tidak cocok untuk Streamlit.** Streamlit membutuhkan server yang terus hidup dan koneksi WebSocket,
sedangkan Vercel menjalankan fungsi serverless jangka pendek (dan PyTorch melebihi batas ukuran fungsinya).
Jika Anda tetap ingin memakai Vercel, itu berarti membangun ulang antarmuka sebagai situs web biasa
(HTML/Next.js) dengan API prediksi di tempat lain, yaitu pekerjaan yang berbeda dari proyek ini.

## Struktur folder

```
app.py                  antarmuka Streamlit
predictor.py            memuat model dan menjalankan prediksi
results_data.py         seluruh angka hasil penelitian (dari output notebook)
export_models_cell.py   sel untuk menyimpan model dari notebook
assets/                 gambar ROC, SHAP, LIME, dll. (diambil dari notebook)
models/                 letakkan 4 file model di sini
requirements.txt
Dockerfile
.streamlit/config.toml
```

## Catatan

Dashboard ini adalah prototipe penelitian, bukan alat diagnosis klinis.
