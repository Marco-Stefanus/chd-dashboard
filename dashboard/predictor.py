"""
Memuat model hasil training (XGBoost, EBM, TabNet) dan menjalankan prediksi
dengan alur yang sama persis seperti fungsi `predict_chd()` di notebook:

- XGBoost & EBM  -> data mentah (raw)
- TabNet         -> data distandarisasi dengan scaler yang sama saat training
- Keputusan akhir -> majority voting (>= 2 dari 3 model = HIGH RISK)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from results_data import FEATURES, MODELS, THRESHOLDS

MODEL_DIR = Path(__file__).parent / "models"

FILES = {
    "XGBoost": MODEL_DIR / "xgb_model.json",
    "EBM": MODEL_DIR / "ebm_model.joblib",
    "TabNet": MODEL_DIR / "tabnet_model.zip",
    "scaler": MODEL_DIR / "scaler.joblib",
}


def load_models() -> tuple[dict, dict]:
    """Muat setiap model secara independen.

    Returns (loaded, errors). Satu model gagal dimuat tidak menghentikan yang lain.
    """
    loaded: dict = {}
    errors: dict = {}

    # XGBoost
    try:
        if not FILES["XGBoost"].exists():
            raise FileNotFoundError(FILES["XGBoost"].name)
        from xgboost import XGBClassifier

        m = XGBClassifier()
        m.load_model(str(FILES["XGBoost"]))
        loaded["XGBoost"] = m
    except Exception as e:  # noqa: BLE001
        errors["XGBoost"] = f"{type(e).__name__}: {e}"

    # EBM
    try:
        if not FILES["EBM"].exists():
            raise FileNotFoundError(FILES["EBM"].name)
        import joblib

        loaded["EBM"] = joblib.load(FILES["EBM"])
    except Exception as e:  # noqa: BLE001
        errors["EBM"] = f"{type(e).__name__}: {e}"

    # Scaler + TabNet (TabNet butuh scaler)
    try:
        for key in ("scaler", "TabNet"):
            if not FILES[key].exists():
                raise FileNotFoundError(FILES[key].name)
        import joblib
        from pytorch_tabnet.tab_model import TabNetClassifier

        scaler = joblib.load(FILES["scaler"])
        tn = TabNetClassifier()
        tn.load_model(str(FILES["TabNet"]))
        loaded["TabNet"] = tn
        loaded["scaler"] = scaler
    except Exception as e:  # noqa: BLE001
        errors["TabNet"] = f"{type(e).__name__}: {e}"

    return loaded, errors


def make_X(patient: dict) -> pd.DataFrame:
    """Satu baris input, urutan kolom = FEATURES (sama persis dengan `predict_chd()` di notebook)."""
    return pd.DataFrame([[patient[f] for f in FEATURES]], columns=FEATURES)


def get_thresholds() -> dict:
    """Pakai thresholds.json hasil notebook (jika ada di models/); jika tidak, nilai dari results_data."""
    import json

    f = MODEL_DIR / "thresholds.json"
    thr = dict(THRESHOLDS)
    if f.exists():
        try:
            d = json.loads(f.read_text())
            thr.update({"XGBoost": d["xgboost"], "EBM": d["ebm"], "TabNet": d["tabnet"]})
        except Exception:  # noqa: BLE001
            pass
    return thr


def predict(patient: dict, loaded: dict) -> pd.DataFrame:
    """Prediksi satu pasien. Hanya model yang berhasil dimuat yang dikembalikan."""
    X = make_X(patient)

    rows = []
    for name in MODELS:
        if name not in loaded:
            continue
        if name == "TabNet":
            Xs = loaded["scaler"].transform(X).astype(np.float32)
            prob = float(loaded["TabNet"].predict_proba(Xs)[0, 1])
        else:
            prob = float(loaded[name].predict_proba(X)[0, 1])
        thr = get_thresholds()[name]
        rows.append(
            {
                "Model": name,
                "Probabilitas": prob,
                "Threshold": thr,
                "Prediksi": int(prob >= thr),
            }
        )
    return pd.DataFrame(rows)


def decide(result: pd.DataFrame) -> tuple[str, int, int]:
    """Majority voting seperti di notebook. Returns (label, votes, n_models)."""
    votes = int(result["Prediksi"].sum())
    n = len(result)
    # Aturan notebook: >= 2 dari 3 model. Jika hanya sebagian model tersedia,
    # gunakan mayoritas sederhana dari model yang tersedia.
    needed = 2 if n == 3 else (n // 2 + 1)
    return ("HIGH" if votes >= needed else "LOW"), votes, n


def explain_xgb(patient: dict, loaded: dict) -> pd.DataFrame | None:
    """Kontribusi tiap fitur (log-odds, metode SHAP bawaan XGBoost) untuk satu pasien."""
    if "XGBoost" not in loaded:
        return None
    import xgboost as xgb

    X = make_X(patient)
    contrib = loaded["XGBoost"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    df = pd.DataFrame({"Fitur": FEATURES, "Kontribusi": contrib[:-1]})
    return df.reindex(df["Kontribusi"].abs().sort_values(ascending=False).index)


def explain_ebm(patient: dict, loaded: dict) -> pd.DataFrame | None:
    """Skor tiap term EBM (log-odds) untuk satu pasien; interaksi ikut ditampilkan."""
    if "EBM" not in loaded:
        return None
    X = make_X(patient)
    d = loaded["EBM"].explain_local(X).data(0)
    df = pd.DataFrame({"Fitur": d["names"], "Kontribusi": d["scores"]})
    return df.reindex(df["Kontribusi"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------- verifikasi vs notebook
def verify_against_notebook(loaded: dict, tol_pp: float = 0.5) -> pd.DataFrame:
    """Jalankan kasus contoh dari notebook lewat model yang dimuat, bandingkan dengan output notebook.

    Selisih (poin persen) > tol_pp berarti file model / scaler / versi library
    di folder ini BUKAN yang dipakai notebook.
    """
    from results_data import EXAMPLE_CASES

    rows = []
    for case, info in EXAMPLE_CASES.items():
        if not info["notebook_prob"]:
            continue
        res = predict(info["inputs"], loaded).set_index("Model")
        for m in MODELS:
            if m not in res.index:
                continue
            app_p = res.loc[m, "Probabilitas"] * 100
            nb_p = info["notebook_prob"][m]
            rows.append({"Kasus": case, "Model": m, "Notebook (%)": nb_p, "App (%)": round(app_p, 2),
                         "Selisih (pp)": round(app_p - nb_p, 2), "Cocok": abs(app_p - nb_p) <= tol_pp})
    return pd.DataFrame(rows)


def env_info(loaded: dict) -> dict:
    """Versi library + info scaler/model untuk melacak ketidakcocokan."""
    import importlib

    info = {}
    for lib in ("xgboost", "interpret", "pytorch_tabnet", "sklearn", "numpy", "pandas", "torch"):
        try:
            info[lib] = importlib.import_module(lib).__version__
        except Exception:  # noqa: BLE001
            info[lib] = "tidak terpasang"
    sc = loaded.get("scaler")
    if sc is not None:
        info["scaler.n_samples_seen_"] = str(getattr(sc, "n_samples_seen_", "?"))
        info["scaler.mean_ (age, sysBP)"] = str([round(float(sc.mean_[FEATURES.index(k)]), 2) for k in ("age", "sysBP")])
        info["scaler.feature_names_in_"] = str(getattr(sc, "feature_names_in_", "None (fit pada array)"))
    return info
