# ============================================================
# JALANKAN SEBAGAI SEL TERAKHIR DI NOTEBOOK ANDA
# (setelah best_xgb, best_ebm, tabnet_model, dan scaler sudah terlatih)
# Hasilnya: folder chd_dashboard/models/ berisi 4 file untuk Streamlit.
# ============================================================
import os, joblib

OUT = "chd_dashboard/models"          # ubah jika folder dashboard ada di tempat lain
os.makedirs(OUT, exist_ok=True)

# 1) XGBoost  -> format JSON bawaan XGBoost (aman lintas versi)
best_xgb.save_model(f"{OUT}/xgb_model.json")

# 2) EBM      -> joblib
joblib.dump(best_ebm, f"{OUT}/ebm_model.joblib")

# 3) Scaler   -> joblib (scaler yang di-fit pada X_train_raw, dipakai TabNet)
joblib.dump(scaler, f"{OUT}/scaler.joblib")

# 4) TabNet   -> otomatis jadi tabnet_model.zip
tabnet_model.save_model(f"{OUT}/tabnet_model")

print("Tersimpan:", sorted(os.listdir(OUT)))
# Harus berisi: ebm_model.joblib, scaler.joblib, tabnet_model.zip, xgb_model.json
