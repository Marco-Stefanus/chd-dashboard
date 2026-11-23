"""
Seluruh angka di file ini disalin dari OUTPUT notebook
`Code_Skripsi_Drop___Threshold_Fixed_v2.ipynb` (evaluasi pada test set, n = 837).
Tidak ada angka yang dihitung ulang atau diperkirakan.
"""

FEATURES = [
    "male", "age", "education", "cigsPerDay", "BPMeds", "prevalentStroke",
    "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI",
    "heartRate", "glucose",
]

MODELS = ["XGBoost", "EBM", "TabNet"]

MODEL_COLORS = {
    "XGBoost": "#0B7A8A",   # teal
    "EBM": "#E8A33D",       # amber
    "TabNet": "#6C63D9",    # violet
}

# Threshold final (FIXED_*_THRESHOLD di notebook), dipilih dengan memaksimalkan F2 pada validation set
THRESHOLDS = {"XGBoost": 0.22, "EBM": 0.20, "TabNet": 0.23}

# ---------------------------------------------------------------- test set
TEST_RESULTS = {
    "XGBoost": dict(accuracy=0.7575, precision=0.6597, recall=0.9382, f1=0.7747,
                    f2=0.8651, roc_auc=0.9252, tn=285, fp=180, fn=23, tp=349),
    "EBM":     dict(accuracy=0.6487, precision=0.5617, recall=0.9543, f1=0.7072,
                    f2=0.8373, roc_auc=0.8933, tn=188, fp=277, fn=17, tp=355),
    "TabNet":  dict(accuracy=0.5639, precision=0.5050, recall=0.9543, f1=0.6605,
                    f2=0.8101, roc_auc=0.7424, tn=117, fp=348, fn=17, tp=355),
}

# ------------------------------------------------- train / val / test (ROC-AUC & lainnya)
OVERFIT_CHECK = {
    "XGBoost": {
        "Train":      dict(accuracy=0.8144, precision=0.7120, recall=0.9781, f2=0.9101, roc_auc=0.9728),
        "Validation": dict(accuracy=0.7694, precision=0.6711, recall=0.9435, f2=0.8727, roc_auc=0.9291),
        "Test":       dict(accuracy=0.7575, precision=0.6597, recall=0.9382, f2=0.8651, roc_auc=0.9252),
    },
    "EBM": {
        "Train":      dict(accuracy=0.6539, precision=0.5650, recall=0.9620, f2=0.8434, roc_auc=0.8981),
        "Validation": dict(accuracy=0.6667, precision=0.5751, recall=0.9570, f2=0.8448, roc_auc=0.8861),
        "Test":       dict(accuracy=0.6487, precision=0.5617, recall=0.9543, f2=0.8373, roc_auc=0.8933),
    },
    "TabNet": {
        "Train":      dict(accuracy=0.5817, precision=0.5158, recall=0.9614, f2=0.8197, roc_auc=0.7628),
        "Validation": dict(accuracy=0.5771, precision=0.5132, recall=0.9435, f2=0.8080, roc_auc=0.7448),
        "Test":       dict(accuracy=0.5639, precision=0.5050, recall=0.9543, f2=0.8101, roc_auc=0.7424),
    },
}

# ---------------------------------------------------------------- hyperparameter terbaik
BEST_PARAMS = {
    "XGBoost": {
        "search": "GridSearchCV, 5-fold Stratified CV, scoring ROC-AUC",
        "cv_score": "CV ROC-AUC = 0.9085",
        "params": {
            "max_depth": 4, "min_child_weight": 5, "gamma": 0.5,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 1, "reg_lambda": 5,
            "learning_rate": 0.05, "n_estimators": 500,
        },
    },
    "EBM": {
        "search": "GridSearchCV, 5-fold Stratified CV, scoring ROC-AUC (216 kandidat)",
        "cv_score": "CV ROC-AUC = 0.8775",
        "params": {
            "interactions": 5, "learning_rate": 0.03, "max_bins": 128,
            "max_interaction_bins": 16, "max_leaves": 2,
            "max_rounds": 200, "min_samples_leaf": 20,
        },
    },
    "TabNet": {
        "search": "Grid search manual (36 kombinasi), seleksi dengan Validation ROC-AUC",
        "cv_score": "Validation ROC-AUC = 0.7448",
        "params": {
            "n_d = n_a": 16, "n_steps": 4, "lambda_sparse": 0.01,
            "weight_decay": 0.0001, "gamma": 1.2, "n_independent": 1, "n_shared": 1,
            "mask_type": "entmax", "lr": 0.001, "scheduler": "StepLR (step 10, gamma 0.9)",
            "max_epochs": 100, "patience": 15, "batch_size": 1024, "virtual_batch_size": 128,
        },
    },
}

# ---------------------------------------------------------------- interpretabilitas
# EBM: global term importance (fitur utama + interaksi)
EBM_IMPORTANCE = [
    ("education", 0.6683), ("age", 0.3783), ("cigsPerDay", 0.3217),
    ("sysBP", 0.1934), ("cigsPerDay & heartRate", 0.1847),
    ("education & BMI", 0.1598), ("heartRate", 0.1486),
    ("heartRate & glucose", 0.1001), ("age & glucose", 0.0953),
    ("BPMeds", 0.0949), ("BMI", 0.0858), ("diaBP", 0.0798),
    ("sysBP & glucose", 0.0753), ("male", 0.0726), ("glucose", 0.0439),
    ("totChol", 0.0415), ("diabetes", 0.0117), ("prevalentStroke", 0.0067),
    ("prevalentHyp", 0.0055),
]

# TabNet: mean |SHAP| (50 sampel test, KernelExplainer)
TABNET_SHAP = [
    ("age", 0.101377), ("cigsPerDay", 0.032561), ("education", 0.029907),
    ("sysBP", 0.029775), ("diaBP", 0.023469), ("heartRate", 0.022601),
    ("male", 0.020150), ("glucose", 0.019004), ("prevalentHyp", 0.018039),
    ("totChol", 0.017405), ("BMI", 0.015491), ("BPMeds", 0.011132),
    ("diabetes", 0.001823), ("prevalentStroke", 0.0),
]

# ---------------------------------------------------------------- dataset
DATASET = {
    "sumber": "Framingham Heart Study (target: TenYearCHD)",
    "sebelum_drop": 4240,
    "setelah_drop": 3658,
    "dihapus": 582,
    "kelas_setelah_drop": {"No CHD": 3101, "CHD": 557},
    "kelas_setelah_balancing": {"No CHD": 3100, "CHD": 2480},
    "split": {"Train": 3906, "Validation": 837, "Test": 837},
    "kelas_test": {"No CHD": 465, "CHD": 372},
}

# ---------------------------------------------------------------- kasus contoh dari notebook
# (input pasien + probabilitas % tiap model, persis seperti output notebook)
EXAMPLE_CASES = {
    "Risiko tinggi · pria (67 th)": dict(
        inputs=dict(male=1, age=67, education=1, cigsPerDay=25, BPMeds=1, prevalentStroke=0,
                    prevalentHyp=1, diabetes=1, totChol=295, sysBP=178, diaBP=102, BMI=34.8,
                    heartRate=92, glucose=165),
        notebook_prob={"XGBoost": 81.52, "EBM": 88.18, "TabNet": 66.71}),
    "Risiko tinggi · wanita (60 th)": dict(
        inputs=dict(male=0, age=60, education=2, cigsPerDay=12, BPMeds=0, prevalentStroke=0,
                    prevalentHyp=1, diabetes=1, totChol=250, sysBP=155, diaBP=92, BMI=29.5,
                    heartRate=84, glucose=135),
        notebook_prob={"XGBoost": 84.52, "EBM": 78.16, "TabNet": 65.04}),
    "Risiko tinggi, 2 dari 3 model · pria (52 th)": dict(
        inputs=dict(male=1, age=52, education=3, cigsPerDay=10, BPMeds=0, prevalentStroke=0,
                    prevalentHyp=1, diabetes=0, totChol=210, sysBP=138, diaBP=86, BMI=26.5,
                    heartRate=75, glucose=95),
        notebook_prob={"XGBoost": 14.26, "EBM": 31.72, "TabNet": 48.71}),
    "Risiko rendah · pria (46 th)": dict(
        inputs=dict(male=1, age=46, education=3, cigsPerDay=2, BPMeds=0, prevalentStroke=0,
                    prevalentHyp=0, diabetes=0, totChol=200, sysBP=125, diaBP=80, BMI=24.0,
                    heartRate=72, glucose=92),
        notebook_prob={"XGBoost": 5.85, "EBM": 18.11, "TabNet": 21.90}),
    "Risiko rendah · wanita (35 th)": dict(
        inputs=dict(male=0, age=35, education=3, cigsPerDay=0, BPMeds=0, prevalentStroke=0,
                    prevalentHyp=0, diabetes=0, totChol=175, sysBP=115, diaBP=72, BMI=23.5,
                    heartRate=66, glucose=80),
        notebook_prob={"XGBoost": 2.00, "EBM": 7.64, "TabNet": 11.74}),
    "Risiko rendah · pria (35 th)": dict(
        inputs=dict(male=1, age=35, education=4, cigsPerDay=0, BPMeds=0, prevalentStroke=0,
                    prevalentHyp=0, diabetes=0, totChol=155, sysBP=105, diaBP=68, BMI=21.5,
                    heartRate=65, glucose=78),
        notebook_prob={"XGBoost": 0.99, "EBM": 2.72, "TabNet": 12.16}),
}

FEATURE_INFO = {
    "male": "Jenis kelamin (1 = pria, 0 = wanita)",
    "age": "Usia (tahun)",
    "education": "Tingkat pendidikan (1-4)",
    "cigsPerDay": "Rokok per hari",
    "BPMeds": "Konsumsi obat tekanan darah",
    "prevalentStroke": "Riwayat stroke",
    "prevalentHyp": "Riwayat hipertensi",
    "diabetes": "Diabetes",
    "totChol": "Kolesterol total (mg/dL)",
    "sysBP": "Tekanan darah sistolik (mmHg)",
    "diaBP": "Tekanan darah diastolik (mmHg)",
    "BMI": "Indeks massa tubuh",
    "heartRate": "Detak jantung (bpm)",
    "glucose": "Glukosa (mg/dL)",
}


# ---------------------------------------------------------------- pemilihan threshold (VALIDATION set)
# Disalin dari output `best_threshold_comparison` di notebook (threshold dengan F2 tertinggi).
VAL_THRESHOLD_RESULTS = {
    "XGBoost": dict(threshold=0.22, accuracy=0.7694, precision=0.6711, recall=0.9435, f1=0.7844, f2=0.8727,
                    tn=293, fp=172, fn=21, tp=351),
    "EBM":     dict(threshold=0.20, accuracy=0.6667, precision=0.5751, recall=0.9570, f1=0.7185, f2=0.8448,
                    tn=202, fp=263, fn=16, tp=356),
    "TabNet":  dict(threshold=0.23, accuracy=0.5771, precision=0.5132, recall=0.9435, f1=0.6648, f2=0.8080,
                    tn=132, fp=333, fn=21, tp=351),
}
