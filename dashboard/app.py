from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import predictor
from watch_sim import watch_html
from results_data import (
    BEST_PARAMS, DATASET, EBM_IMPORTANCE, EXAMPLE_CASES, FEATURE_INFO, FEATURES,
    MODEL_COLORS, MODELS, OVERFIT_CHECK, TABNET_SHAP, TEST_RESULTS, THRESHOLDS,
    VAL_THRESHOLD_RESULTS,
)

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="CHD Risk · XGBoost, EBM, TabNet",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ desain
INK = "#0F2A3D"
MUTED = "#5A7184"
LINE = "#DCE8EE"
PRIMARY = "#0B7A8A"
PRIMARY_DARK = "#075E6B"
PRIMARY_SOFT = "#E3F3F6"
RISK = "#D64545"
RISK_BG = "#FDECEC"
SAFE = "#1F9D6B"
SAFE_BG = "#E5F6EE"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --ink:#0F2A3D; --muted:#5A7184; --line:#DCE8EE; --primary:#0B7A8A; --primary-dark:#075E6B;
  --primary-soft:#E3F3F6; --risk:#D64545; --risk-bg:#FDECEC; --safe:#1F9D6B; --safe-bg:#E5F6EE;
}

html, body, .stApp, [class*="css"] { font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #F3F8FA; color: var(--ink); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4 { color: var(--ink); font-family: 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -0.01em; }
h3 { font-weight: 700 !important; font-size: 1.15rem !important; }
h4 { font-weight: 700 !important; font-size: 1.02rem !important; margin: .3rem 0 .4rem; padding: .3rem 0 0; }
p, li { line-height: 1.65; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.4rem; }
.brand-logo { width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg,#075E6B,#19A7B5);
  display:flex; align-items:center; justify-content:center; box-shadow:0 4px 12px rgba(11,122,138,.28); }
.brand-name { font-weight:800; font-size:1.12rem; color:var(--ink); line-height:1.1; }
.brand-sub { font-size:.78rem; color:var(--muted); }

[data-testid="stSidebar"] [role="radiogroup"] { gap:.25rem; }
/* DOM baru */
[data-testid="stSidebar"] [data-testid="stRadioOption"] { padding:.6rem .85rem; border-radius:10px; width:100%; margin:0; cursor:pointer; transition:background .15s; }
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background:#F0F6F8; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font-size:.95rem; font-weight:500; color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] { background:var(--primary-soft); box-shadow:inset 3px 0 0 var(--primary); }
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p { color:var(--primary-dark); font-weight:700; }
/* DOM lama */
[data-testid="stSidebar"] label[data-baseweb="radio"] { padding:.6rem .85rem; border-radius:10px; width:100%; margin:0; transition:background .15s; }
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background:#F0F6F8; }
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display:none; }
[data-testid="stSidebar"] label[data-baseweb="radio"] p { font-size:.95rem; font-weight:500; color:var(--ink); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) { background:var(--primary-soft); box-shadow:inset 3px 0 0 var(--primary); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) p { color:var(--primary-dark); font-weight:700; }

/* ---------- hero ---------- */
.hero { position:relative; overflow:hidden; border-radius:18px; padding:1.9rem 2.1rem; margin-bottom:1.4rem;
  background:linear-gradient(120deg,#075E6B 0%,#0B7A8A 55%,#19A7B5 100%); color:#fff;
  box-shadow:0 10px 28px rgba(7,94,107,.22); }
.hero h1 { color:#fff !important; font-size:1.95rem; font-weight:800; margin:.15rem 0 .45rem; line-height:1.2; padding:0; }
.hero p { color:rgba(255,255,255,.88); font-size:1.02rem; max-width:60ch; margin:0; }
.hero .tag { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
  padding:.18rem .75rem; border-radius:999px; font-size:.78rem; font-weight:600; letter-spacing:.02em; }
.hero svg { position:absolute; right:-10px; bottom:6px; width:46%; opacity:.28; }

/* ---------- section ---------- */
.sec { display:flex; align-items:center; gap:.6rem; margin:1.7rem 0 .8rem; }
.sec .bar { width:5px; height:22px; border-radius:4px; background:var(--primary); }
.sec .t { font-size:1.18rem; font-weight:800; color:var(--ink); }
.sec-desc { color:var(--muted); font-size:.93rem; margin:-.4rem 0 .9rem; max-width:75ch; }

/* ---------- cards ---------- */
[class*="st-key-card-"] { background:#fff; border:1px solid var(--line); border-radius:16px;
  box-shadow:0 2px 10px rgba(15,42,61,.04); padding:1.05rem 1.2rem; }
.mc { background:#fff; border:1px solid var(--line); border-top:4px solid var(--c); border-radius:16px; padding:1.1rem 1.25rem 1rem;
  box-shadow:0 2px 10px rgba(15,42,61,.05); height:100%; }
.mc-head { display:flex; align-items:center; gap:.5rem; }
.mc-name { font-weight:800; font-size:1.1rem; color:var(--ink); }
.mc-badge { margin-left:auto; font-size:.7rem; font-weight:700; padding:.18rem .6rem; border-radius:999px;
  background:var(--primary-soft); color:var(--primary-dark); }
.mc-label { color:var(--muted); font-size:.82rem; margin-top:.9rem; }
.mc-value { font-size:2.35rem; font-weight:800; color:var(--ink); line-height:1.1; margin:.1rem 0 .9rem; }
.mc-stats { display:flex; gap:.5rem; border-top:1px solid var(--line); padding-top:.75rem; }
.mc-stats div { flex:1; }
.mc-stats b { display:block; font-size:1.02rem; color:var(--ink); }
.mc-stats span { font-size:.75rem; color:var(--muted); }

.insight { min-height:132px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem; height:100%;
  display:flex; gap:.9rem; align-items:flex-start; box-shadow:0 2px 10px rgba(15,42,61,.04); }
.insight .ic { flex:0 0 40px; height:40px; border-radius:12px; background:var(--primary-soft); display:flex;
  align-items:center; justify-content:center; font-size:1.25rem; }
.insight .tt { font-weight:700; color:var(--ink); margin-bottom:.15rem; font-size:.98rem; }
.insight .tx { color:var(--muted); font-size:.9rem; line-height:1.55; }

.gloss { background:var(--primary-soft); border-radius:12px; padding:.85rem 1rem; height:100%; min-height:118px; }
.gloss b { color:var(--primary-dark); font-size:.95rem; }
.gloss p { margin:.25rem 0 0; font-size:.85rem; color:var(--ink); line-height:1.5; }

.callout { display:flex; gap:.75rem; border-radius:12px; padding:.85rem 1.05rem; margin:.6rem 0; font-size:.92rem; line-height:1.55; }
.callout.info { background:#E8F1FB; color:#16436F; border:1px solid #CFE0F3; }
.callout.warn { background:#FFF6E0; color:#7A4F00; border:1px solid #F4E1AE; }
.callout .ci { font-size:1.15rem; line-height:1.4; }

.note { color:var(--muted); font-size:.9rem; line-height:1.6; }

/* ---------- hasil prediksi ---------- */
.verdict { display:flex; align-items:center; gap:1rem; border-radius:16px; padding:1.15rem 1.3rem;
  background:var(--bg); border:1.5px solid var(--c); }
.v-icon { width:54px; height:54px; border-radius:50%; background:var(--c); color:#fff; display:flex;
  align-items:center; justify-content:center; font-size:1.6rem; font-weight:800; flex:0 0 54px; }
.v-body { flex:1; }
.v-title { font-size:1.45rem; font-weight:800; color:var(--c); line-height:1.15; }
.v-sub { color:var(--ink); font-size:.9rem; margin-top:.2rem; }
.v-score { text-align:center; padding-left:1rem; border-left:1px solid rgba(15,42,61,.12); }
.v-score b { font-size:2rem; font-weight:800; color:var(--c); }
.v-score span { font-size:1.1rem; color:var(--muted); font-weight:600; }
.v-score small { display:block; color:var(--muted); font-size:.72rem; }
.chip { display:inline-block; margin:.7rem .4rem 0 0; padding:.28rem .85rem; border-radius:999px; font-size:.84rem; font-weight:600; }

.step { display:flex; gap:.9rem; align-items:flex-start; padding:.7rem 0; border-bottom:1px dashed var(--line); }
.step:last-child { border-bottom:none; }
.step .n { flex:0 0 30px; height:30px; border-radius:50%; background:var(--primary); color:#fff; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-size:.85rem; }
.step .s { font-size:.93rem; line-height:1.55; }

.stat { text-align:center; padding:.9rem .5rem; min-height:84px; }
.stat b { display:block; font-size:1.7rem; font-weight:800; color:var(--primary-dark); }
.stat span { color:var(--muted); font-size:.82rem; }

.pipe { display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin:.2rem 0 .4rem; }
.pipe .b { background:var(--primary-soft); border-radius:10px; padding:.45rem .75rem; font-size:.82rem; color:var(--primary-dark); font-weight:600; line-height:1.3; }
.pipe .a { color:var(--muted); font-weight:700; }

/* ---------- widget ---------- */
.stButton > button { border-radius:10px; }
div[data-baseweb="select"] > div, .stNumberInput input { border-radius:10px !important; }
button[data-baseweb="tab"] { font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--primary-dark); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:var(--primary); }
[data-testid="stMetric"] { background:#F7FBFC; border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem; }
[data-testid="stMetricLabel"] p { color:var(--muted); font-size:.82rem; }
[data-testid="stMetricValue"] { color:var(--primary-dark); font-weight:800; }
.stCaption, [data-testid="stCaptionContainer"] { color:var(--muted); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ komponen UI
ECG = (
    '<svg viewBox="0 0 400 90" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M0 50 H70 L88 50 L100 14 L118 84 L134 36 L146 50 H230 L248 50 '
    'L260 14 L278 84 L294 36 L306 50 H400"/></svg>'
)
LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s-7-4.6-9.2-9.2C1.1 8.2 3 4.5 6.7 4.5c2 0 3.5 1 '
    '5.3 3.2 1.8-2.2 3.3-3.2 5.3-3.2 3.7 0 5.6 3.7 3.9 7.3C19 16.4 12 21 12 21z"/>'
    '<path d="M6 12h3l1.5-2.5 2 5 1.5-2.5H18"/></svg>'
)


def hero(title: str, subtitle: str, tag: str = "Framingham Heart Study · Risiko CHD 10 tahun") -> None:
    st.markdown(
        f'<div class="hero">{ECG}<span class="tag">{tag}</span><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, desc: str | None = None) -> None:
    st.markdown(f'<div class="sec"><div class="bar"></div><div class="t">{title}</div></div>',
                unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info", icon: str | None = None) -> None:
    icon = icon or ("ℹ️" if kind == "info" else "⚠️")
    st.markdown(f'<div class="callout {kind}"><div class="ci">{icon}</div><div>{text}</div></div>',
                unsafe_allow_html=True)


_CARD_IDS = itertools.count()  # di-reset setiap rerun karena skrip dieksekusi ulang dari atas


def render_html(html: str, height: int) -> None:
    """Tampilkan HTML interaktif. st.iframe di Streamlit baru; components.html di versi lama."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def card():
    """Kartu putih. Gaya diterapkan lewat CSS pada class `st-key-card-*` (Streamlit >= 1.39)."""
    return st.container(key=f"card-{next(_CARD_IDS)}")


def show_plot(fig: go.Figure) -> None:
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(size=15, color=INK), x=0.01),
        hoverlabel=dict(font_family="Plus Jakarta Sans, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#E9F0F4", zerolinecolor="#E9F0F4")
    fig.update_yaxes(gridcolor="#E9F0F4", zerolinecolor="#E9F0F4")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True)


def show_image(name: str, caption: str | None = None) -> None:
    path = ASSETS / name
    if not path.exists():
        st.warning(f"Gambar {name} tidak ditemukan.")
        return
    try:
        st.image(str(path), caption=caption, width="stretch")
    except TypeError:
        st.image(str(path), caption=caption, use_container_width=True)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


@st.cache_resource(show_spinner="Memuat model…")
def get_models():
    return predictor.load_models()


def results_table() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        r = TEST_RESULTS[m]
        rows.append({"Model": m, "Threshold": THRESHOLDS[m], "Recall": r["recall"], "F2": r["f2"],
                     "ROC-AUC": r["roc_auc"], "Accuracy": r["accuracy"], "Precision": r["precision"],
                     "F1": r["f1"]})
    return pd.DataFrame(rows)


def model_card(m: str, badge: str | None) -> None:
    r = TEST_RESULTS[m]
    b = f'<span class="mc-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""<div class="mc" style="--c:{MODEL_COLORS[m]}">
        <div class="mc-head"><span class="mc-name">{m}</span>{b}</div>
        <div class="mc-label">Recall pada test set (metrik utama)</div>
        <div class="mc-value">{pct(r['recall'])}</div>
        <div class="mc-stats">
          <div><b>{r['f2']:.3f}</b><span>F2</span></div>
          <div><b>{pct(r['precision'])}</b><span>Precision</span></div>
          <div><b>{r['roc_auc']:.3f}</b><span>ROC-AUC</span></div>
        </div></div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ halaman
def page_summary() -> None:
    hero("Prediksi Risiko Penyakit Jantung Koroner",
         "Tiga model machine learning dibandingkan untuk menandai pasien yang berisiko CHD dalam 10 tahun. "
         "Ambang keputusan diatur agar sedikit kasus CHD yang terlewat.")

    top_rec = max(round(TEST_RESULTS[m]["recall"], 4) for m in MODELS)
    best_auc = max(MODELS, key=lambda m: TEST_RESULTS[m]["roc_auc"])
    cols = st.columns(3, gap="medium")
    for col, m in zip(cols, MODELS):
        badges = []
        if m == "EBM":
            badges.append("Model terbaik")
        if round(TEST_RESULTS[m]["recall"], 4) == top_rec:
            badges.append("Recall tertinggi")
        if m == best_auc:
            badges.append("ROC-AUC tertinggi")
        with col:
            model_card(m, " · ".join(badges) if badges else None)

    x, e, t = TEST_RESULTS["XGBoost"], TEST_RESULTS["EBM"], TEST_RESULTS["TabNet"]
    section("Temuan utama")
    c1, c2 = st.columns(2, gap="medium")
    insights = [
        ("🏆", "EBM adalah model terbaik",
         f"Recall {pct(e['recall'])}: hanya {e['fn']} dari {e['fn'] + e['tp']} pasien CHD yang terlewat, "
         f"lebih sedikit dari XGBoost ({x['fn']}). Dalam skrining medis, melewatkan pasien CHD lebih berbahaya "
         f"daripada alarm palsu, sehingga recall menjadi metrik utama."),
        ("⚖️", "XGBoost unggul di precision dan ROC-AUC, tetapi recall kalah",
         f"Recall {pct(x['recall'])} ({x['fn']} terlewat) lebih rendah dari EBM, namun precision {pct(x['precision'])}, "
         f"ROC-AUC {x['roc_auc']:.4f}, dan F2 {x['f2']:.4f} lebih tinggi. Dalam medis, recall tetap yang "
         f"paling penting, jadi keunggulan ini tidak mengalahkan EBM."),
        ("📉", "TabNet paling lemah",
         f"Recall {pct(t['recall'])} sama dengan EBM ({t['fn']} terlewat), tetapi ROC-AUC hanya {t['roc_auc']:.4f}, "
         f"precision {pct(t['precision'])}, dan {t['fp']} alarm palsu (EBM {e['fp']}). Recall setara, kualitas "
         f"pemisahan kelasnya jauh di bawah."),
        ("🩺", "Harga dari recall yang tinggi",
         f"Ketiganya mencapai recall di atas 93%. Dibanding XGBoost, EBM menemukan {x['fn'] - e['fn']} pasien CHD "
         f"lebih banyak, dengan tambahan {e['fp'] - x['fp']} alarm palsu ({e['fp']} vs {x['fp']})."),
    ]
    for i, (ic, tt, tx) in enumerate(insights):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f'<div class="insight"><div class="ic">{ic}</div><div><div class="tt">{tt}</div>'
                        f'<div class="tx">{tx}</div></div></div><div style="height:.8rem"></div>',
                        unsafe_allow_html=True)

    section("Hasil lengkap pada test set",
            f"{DATASET['split']['Test']} pasien ({DATASET['kelas_test']['CHD']} CHD, "
            f"{DATASET['kelas_test']['No CHD']} tanpa CHD). Sel hijau = nilai tertinggi pada kolom.")
    df = results_table()
    metric_cols = [c for c in df.columns if c not in ("Model", "Threshold")]
    with card():
        st.dataframe(
            df.style.format({c: "{:.4f}" for c in metric_cols} | {"Threshold": "{:.2f}"})
            .highlight_max(subset=metric_cols, color="#CDEBDD"),
            hide_index=True, width="stretch",
        )

    section("Cara membaca metrik", "Penjelasan singkat untuk pembaca yang belum familiar dengan istilahnya.")
    g = st.columns(4, gap="medium")
    gloss = [
        ("Recall", "Dari semua pasien yang benar-benar CHD, berapa persen yang berhasil terdeteksi."),
        ("Precision", "Dari semua pasien yang ditandai CHD, berapa persen yang memang CHD."),
        ("F2 score", "Skor gabungan yang lebih mementingkan recall dibanding precision."),
        ("ROC-AUC", "Kemampuan membedakan pasien CHD dan non-CHD. 1,0 sempurna, 0,5 setara menebak."),
    ]
    for col, (n, d) in zip(g, gloss):
        col.markdown(f'<div class="gloss"><b>{n}</b><p>{d}</p></div>', unsafe_allow_html=True)


def page_compare() -> None:
    hero("Perbandingan Model", "Seberapa baik tiap model mendeteksi CHD, dan berapa harga yang dibayar "
                               "dalam bentuk alarm palsu.")

    section("Metrik pada test set",
            "Recall adalah metrik utama: dalam skrining CHD, pasien yang terlewat lebih berbahaya daripada alarm palsu.")
    metrics = ["Recall", "F2", "Precision", "ROC-AUC", "Accuracy", "F1"]
    df = results_table()
    fig = go.Figure()
    for m in MODELS:
        row = df[df.Model == m].iloc[0]
        fig.add_bar(name=m, x=metrics, y=[row[k] for k in metrics], marker_color=MODEL_COLORS[m],
                    text=[f"{row[k]:.3f}" for k in metrics], textposition="outside")
    fig.update_layout(barmode="group", bargap=0.25, yaxis=dict(range=[0, 1.1], title="Nilai"),
                      legend=dict(orientation="h", y=1.14), height=420)
    with card():
        show_plot(fig)

    e, x, t = TEST_RESULTS["EBM"], TEST_RESULTS["XGBoost"], TEST_RESULTS["TabNet"]
    callout(f"<b>Berdasarkan recall:</b> EBM dan TabNet sama-sama {pct(e['recall'])} ({e['fn']} pasien CHD terlewat), "
            f"XGBoost {pct(x['recall'])} ({x['fn']} terlewat). Di antara dua model dengan recall sama, EBM lebih baik "
            f"pada F2 ({e['f2']:.3f} vs {t['f2']:.3f}), precision, ROC-AUC, dan alarm palsu ({e['fp']} vs {t['fp']}). "
            f"XGBoost unggul di precision, ROC-AUC, dan F2, tetapi recall tetap yang utama.", "info", "🎯")

    section("Kasus terlewat vs alarm palsu",
            "Threshold rendah membuat model lebih memilih salah menandai pasien sehat daripada "
            "melewatkan pasien CHD. Grafik ini menunjukkan harga dari pilihan itu.")
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "CHD yang terlewat (False Negative)", "Pasien sehat ditandai CHD (False Positive)"))
    for m in MODELS:
        r = TEST_RESULTS[m]
        fig.add_bar(x=[m], y=[r["fn"]], marker_color=MODEL_COLORS[m], showlegend=False,
                    text=[r["fn"]], textposition="outside", row=1, col=1)
        fig.add_bar(x=[m], y=[r["fp"]], marker_color=MODEL_COLORS[m], showlegend=False,
                    text=[r["fp"]], textposition="outside", row=1, col=2)
    fig.update_yaxes(range=[0, 380])
    fig.update_layout(height=340)
    with card():
        show_plot(fig)

    section("Pemilihan threshold (validation set)",
            "Threshold dipilih dengan F2 tertinggi pada validation set (rentang 0,20 sampai 0,99, langkah 0,01; "
            "jika F2 sama dipilih threshold yang lebih tinggi), lalu dipakai tetap pada test set.")
    vt = pd.DataFrame([{"Model": m, "Threshold": v["threshold"], "Recall": v["recall"], "F2": v["f2"],
                        "Precision": v["precision"], "F1": v["f1"], "Accuracy": v["accuracy"],
                        "TP": v["tp"], "FN": v["fn"], "FP": v["fp"], "TN": v["tn"]}
                       for m, v in VAL_THRESHOLD_RESULTS.items()])
    with card():
        st.dataframe(vt.style.format({"Threshold": "{:.2f}", "Recall": "{:.4f}", "F2": "{:.4f}",
                                      "Precision": "{:.4f}", "F1": "{:.4f}", "Accuracy": "{:.4f}"}),
                     hide_index=True, width="stretch")

    section("Confusion matrix (test set)")
    cols = st.columns(3, gap="medium")
    for col, m in zip(cols, MODELS):
        r = TEST_RESULTS[m]
        z = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        f = go.Figure(go.Heatmap(
            z=z, x=["No CHD", "CHD"], y=["No CHD", "CHD"],
            colorscale=[[0, "#F1F6F9"], [1, MODEL_COLORS[m]]], showscale=False,
            text=z, texttemplate="%{text}", textfont=dict(size=22),
        ))
        f.update_layout(title=f"{m} (threshold {THRESHOLDS[m]:.2f})", xaxis_title="Prediksi",
                        yaxis_title="Aktual", yaxis=dict(autorange="reversed"), height=340)
        with col:
            with card():
                show_plot(f)

    section("Kurva ROC")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        with card():
            show_image("roc_curve.png")
    with c2:
        callout("Kurva <b>XGBoost</b> paling dekat ke sudut kiri-atas, artinya ia paling baik memisahkan "
                "pasien CHD dan non-CHD di semua threshold. Kurva <b>TabNet</b> paling dekat ke garis acak.")

    section("Pemeriksaan overfitting",
            "Membandingkan ROC-AUC pada train, validation, dan test. Selisih kecil berarti model stabil.")
    rows = []
    for m in MODELS:
        for split in ("Train", "Validation", "Test"):
            rows.append({"Model": m, "Data": split, **OVERFIT_CHECK[m][split]})
    od = pd.DataFrame(rows)
    labels = {"recall": "Recall", "f2": "F2", "precision": "Precision", "roc_auc": "ROC-AUC", "accuracy": "Accuracy"}
    sel = st.radio("Metrik", list(labels.values()), horizontal=True, index=0, label_visibility="collapsed")
    key = next(k for k, v in labels.items() if v == sel)
    fig = go.Figure()
    shades = {"Train": 0.4, "Validation": 0.7, "Test": 1.0}
    for split in ("Train", "Validation", "Test"):
        sub = od[od.Data == split]
        fig.add_bar(name=split, x=sub.Model, y=sub[key],
                    marker_color=[MODEL_COLORS[m] for m in sub.Model], marker_opacity=shades[split],
                    text=[f"{v:.3f}" for v in sub[key]], textposition="outside")
    fig.update_layout(barmode="group", yaxis=dict(range=[0.4, 1.05], title=sel), height=380,
                      title=f"{sel} train, validation, test (pudar → pekat)")
    with card():
        show_plot(fig)
    gap_a = {m: OVERFIT_CHECK[m]["Train"]["roc_auc"] - OVERFIT_CHECK[m]["Test"]["roc_auc"] for m in MODELS}
    gap_r = {m: OVERFIT_CHECK[m]["Train"]["recall"] - OVERFIT_CHECK[m]["Test"]["recall"] for m in MODELS}
    callout(f"Selisih recall train − test: <b>XGBoost {gap_r['XGBoost']:.4f}</b>, <b>EBM {gap_r['EBM']:.4f}</b>, "
            f"<b>TabNet {gap_r['TabNet']:.4f}</b>. Selisih ROC-AUC train − test: <b>XGBoost {gap_a['XGBoost']:.4f}</b>, "
            f"<b>EBM {gap_a['EBM']:.4f}</b>, <b>TabNet {gap_a['TabNet']:.4f}</b>. EBM paling stabil. "
            "XGBoost selisihnya terbesar tetapi tetap memiliki ROC-AUC test tertinggi.")
    with st.expander("Lihat tabel lengkap"):
        st.dataframe(
            od.rename(columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                               "f2": "F2", "roc_auc": "ROC-AUC"})
            .style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F2", "ROC-AUC"]}),
            hide_index=True, width="stretch",
        )


def page_detail() -> None:
    hero("Detail per Model", "Hyperparameter terbaik dan hasil evaluasi masing-masing model.")
    desc = {
        "XGBoost": "Gradient boosting pada pohon keputusan. Bekerja langsung pada data mentah.",
        "EBM": "Explainable Boosting Machine: model aditif (glass-box) dengan 5 interaksi berpasangan. "
               "Bekerja pada data mentah.",
        "TabNet": "Jaringan saraf berbasis attention untuk data tabular. "
                  "Memakai data yang sudah distandarisasi (StandardScaler).",
    }
    tabs = st.tabs(MODELS)
    for tab, m in zip(tabs, MODELS):
        with tab:
            r = TEST_RESULTS[m]
            callout(desc[m], "info", "🧠")
            c = st.columns(4)
            c[0].metric("Threshold", f"{THRESHOLDS[m]:.2f}")
            c[1].metric("Recall", pct(r["recall"]))
            c[2].metric("ROC-AUC", f"{r['roc_auc']:.4f}")
            c[3].metric("F2", f"{r['f2']:.4f}")
            left, right = st.columns([3, 2], gap="medium")
            with left:
                with card():
                    st.markdown("**Hyperparameter terbaik**")
                    p = BEST_PARAMS[m]
                    st.dataframe(pd.DataFrame({"Parameter": list(p["params"].keys()),
                                               "Nilai": [str(v) for v in p["params"].values()]}),
                                 hide_index=True, width="stretch")
                    st.caption(f"{p['search']}. {p['cv_score']}.")
            with right:
                with card():
                    st.markdown("**Hasil test set**")
                    st.dataframe(
                        pd.DataFrame({
                            "Metrik": ["Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC"],
                            "Nilai": [r["accuracy"], r["precision"], r["recall"], r["f1"], r["f2"], r["roc_auc"]],
                        }).style.format({"Nilai": "{:.4f}"}),
                        hide_index=True, width="stretch")
                    st.caption(f"TN {r['tn']} · FP {r['fp']} · FN {r['fn']} · TP {r['tp']}")


def importance_chart(items, title, color, highlight_interactions=False) -> go.Figure:
    names = [n for n, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colors = [("#F2C46D" if "&" in n else color) if highlight_interactions else color for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.4f}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, height=max(380, 26 * len(items) + 90), xaxis=dict(range=[0, max(vals) * 1.2]))
    return fig


def page_xai() -> None:
    hero("Interpretabilitas Model",
         "Faktor apa yang paling memengaruhi prediksi? SHAP dan LIME menjelaskan model black-box, "
         "sedangkan EBM dapat dibaca langsung.")
    tabs = st.tabs(MODELS)

    with tabs[0]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card():
                st.markdown("**SHAP summary (beeswarm)**")
                show_image("shap_xgb_beeswarm.png")
        with c2:
            with card():
                st.markdown("**SHAP rata-rata |nilai|**")
                show_image("shap_xgb_bar.png")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            with card():
                st.markdown("**Ketergantungan terhadap usia**")
                show_image("shap_xgb_age.png")
        with c4:
            with card():
                st.markdown("**Penjelasan satu pasien (waterfall)**")
                show_image("shap_xgb_waterfall.png")
        with card():
            st.markdown("**LIME, satu pasien uji**")
            show_image("lime_xgb.png")

    with tabs[1]:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            with card():
                show_plot(importance_chart(EBM_IMPORTANCE, "Global term importance, EBM",
                                           MODEL_COLORS["EBM"], highlight_interactions=True))
        with c2:
            callout("Batang <b>kuning muda</b> menandai <b>interaksi dua fitur</b> yang ditemukan EBM; "
                    "batang <b>oranye</b> adalah efek satu fitur. Skala nilai ini khusus EBM dan tidak "
                    "sebanding langsung dengan nilai SHAP model lain.")
        with st.expander("Gambar asli dari notebook"):
            show_image("ebm_importance.png")

    with tabs[2]:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            with card():
                show_plot(importance_chart(TABNET_SHAP, "Mean |SHAP|, TabNet", MODEL_COLORS["TabNet"]))
        with c2:
            callout("Dihitung pada 50 sampel test dengan KernelExplainer. Usia mendominasi dengan selisih "
                    "jauh dari fitur lain. Karena sampelnya kecil, baca sebagai gambaran, bukan angka pasti.")
        c3, c4 = st.columns(2, gap="medium")
        with c3:
            with card():
                st.markdown("**SHAP waterfall, satu pasien**")
                show_image("shap_tabnet_waterfall.png")
        with c4:
            with card():
                st.markdown("**LIME, satu pasien uji**")
                show_image("lime_tabnet.png")


# ------------------------------------------------------------------ prediksi pasien
BINARY = ["BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
DEFAULT_INPUTS = dict(male=0, age=55, education=2, cigsPerDay=0, BPMeds=False,
                      prevalentStroke=False, prevalentHyp=False, diabetes=False,
                      totChol=220, sysBP=130, diaBP=82, BMI=26.0, heartRate=75, glucose=90)
EDU = {1: "1 · Higher secondary", 2: "2 · Graduation", 3: "3 · Post graduation", 4: "4 · PhD"}
FALLBACK_CASE = "Risiko tinggi · pria (67 th)"


def apply_preset() -> None:
    name = st.session_state.get("preset")
    if name in EXAMPLE_CASES:
        for k, v in EXAMPLE_CASES[name]["inputs"].items():
            st.session_state[f"in_{k}"] = bool(v) if k in BINARY else v


def probability_figure(result: pd.DataFrame) -> go.Figure:
    result = result.iloc[::-1]
    colors = [MODEL_COLORS[m] if p == 1 else "#C3CFD8" for m, p in zip(result["Model"], result["Prediksi"])]
    fig = go.Figure()
    fig.add_bar(y=result["Model"], x=result["Probabilitas"] * 100, orientation="h", marker_color=colors,
                width=0.5, text=[f"{v * 100:.1f}%" for v in result["Probabilitas"]],
                textposition="outside", cliponaxis=False, name="Probabilitas CHD")
    fig.add_trace(go.Scatter(
        y=result["Model"], x=result["Threshold"] * 100, mode="markers",
        marker=dict(symbol="line-ns", size=40, line=dict(width=3, color=INK)),
        name="Threshold", hovertemplate="Threshold %{x:.0f}%<extra></extra>"))
    fig.update_layout(height=250, xaxis=dict(range=[0, 110], title="Probabilitas CHD (%)"),
                      legend=dict(orientation="h", y=1.22), margin=dict(l=10, r=40, t=40, b=10))
    return fig


def contribution_figure(ex: pd.DataFrame, model: str) -> go.Figure:
    ex = ex.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ex["Kontribusi"], y=ex["Fitur"], orientation="h",
        marker_color=[RISK if v > 0 else SAFE for v in ex["Kontribusi"]],
        text=[f"{v:+.2f}" for v in ex["Kontribusi"]], textposition="outside", cliponaxis=False))
    fig.update_layout(title=f"{model}: 10 faktor terkuat", height=360, xaxis_title="Kontribusi (log-odds)")
    return fig


def render_result(result: pd.DataFrame, label: str, votes: int, n: int, source: str) -> None:
    high = label == "HIGH"
    color, bg = (RISK, RISK_BG) if high else (SAFE, SAFE_BG)
    title = "Risiko CHD Tinggi" if high else "Risiko CHD Rendah"
    icon = "!" if high else "✓"
    st.markdown(
        f"""<div class="verdict" style="--c:{color};--bg:{bg}">
        <div class="v-icon">{icon}</div>
        <div class="v-body"><div class="v-title">{title}</div>
        <div class="v-sub">{votes} dari {n} model menandai pasien ini sebagai CHD. {source}</div></div>
        <div class="v-score"><b>{votes}</b><span>/{n}</span><small>suara model</small></div></div>""",
        unsafe_allow_html=True,
    )
    chips = "".join(
        f'<span class="chip" style="background:{MODEL_COLORS[r.Model] if r.Prediksi else "#E6EDF2"};'
        f'color:{"#fff" if r.Prediksi else MUTED}">{r.Model}: {"CHD" if r.Prediksi else "No CHD"}</span>'
        for r in result.itertuples()
    )
    st.markdown(chips, unsafe_allow_html=True)
    show_plot(probability_figure(result))
    st.caption("Batang berwarna = model menandai CHD (probabilitas ≥ threshold). Garis hitam = threshold model. "
               "Hasil hanya HIGH atau LOW (tidak ada kategori sedang): HIGH jika ≥ 2 dari 3 model menandai CHD. "
               "Threshold sengaja rendah agar kasus CHD tidak terlewat, sehingga probabilitas ±20% saja "
               "sudah bisa ditandai CHD.")


def page_predict() -> None:
    hero("Prediksi Pasien", "Isi data pasien, lalu lihat prediksi dari tiga model beserta faktor "
                            "yang memengaruhinya.", tag="Simulasi prediksi · Bukan alat diagnosis")
    loaded, errors = get_models()
    available = [m for m in MODELS if m in loaded]
    for k, v in DEFAULT_INPUTS.items():
        st.session_state.setdefault(f"in_{k}", v)

    with card():
        a, b = st.columns([2, 3], gap="large")
        a.markdown("**Mulai cepat dengan contoh kasus**")
        a.caption("Memilih contoh akan mengisi semua kolom di bawah. Anda tetap bisa mengubahnya.")
        b.selectbox("Contoh kasus", ["Isi manual"] + list(EXAMPLE_CASES), key="preset",
                    on_change=apply_preset, label_visibility="collapsed")

    left, right = st.columns([5, 6], gap="large")

    with left:
        with card():
            st.markdown("#### 👤 Profil pasien")
            a, b = st.columns(2)
            male = a.selectbox("Jenis kelamin", [0, 1], key="in_male",
                               format_func=lambda v: "Pria" if v else "Wanita")
            age = b.number_input("Usia (tahun)", 20, 90, step=1, key="in_age",
                                 help="Data Framingham mencakup usia sekitar 32–70 tahun.")
            education = a.selectbox("Pendidikan", [1, 2, 3, 4], key="in_education",
                                    format_func=lambda v: EDU[v])
            cigs = b.number_input("Rokok per hari", 0, 70, step=1, key="in_cigsPerDay",
                                  help="0 jika tidak merokok.")

            st.markdown("#### 🩺 Riwayat penyakit")
            t1, t2 = st.columns(2)
            hyp = t1.toggle("Riwayat hipertensi", key="in_prevalentHyp")
            bpm = t1.toggle("Obat tekanan darah", key="in_BPMeds")
            diab = t2.toggle("Diabetes", key="in_diabetes")
            stroke = t2.toggle("Riwayat stroke", key="in_prevalentStroke")
            st.caption("Menyala = Ya, mati = Tidak.")

            st.markdown("#### 🔬 Hasil pemeriksaan")
            a, b = st.columns(2)
            sys_bp = a.number_input("Sistolik (mmHg)", 70, 300, step=1, key="in_sysBP",
                                    help="Normal: di bawah 120 mmHg.")
            dia_bp = b.number_input("Diastolik (mmHg)", 40, 160, step=1, key="in_diaBP",
                                    help="Normal: di bawah 80 mmHg.")
            chol = a.number_input("Kolesterol total (mg/dL)", 100, 700, step=1, key="in_totChol",
                                  help="Diinginkan: di bawah 200 mg/dL.")
            glucose = b.number_input("Glukosa (mg/dL)", 40, 400, step=1, key="in_glucose",
                                     help="Glukosa puasa normal: 70–99 mg/dL.")
            bmi = a.number_input("BMI", 14.0, 60.0, step=0.1, key="in_BMI",
                                 help="Normal: 18,5–24,9.")
            hr = b.number_input("Detak jantung (bpm)", 40, 150, step=1, key="in_heartRate",
                                help="Detak jantung istirahat normal: 60–100 bpm.")

            if age < 32 or age > 70:
                callout("Usia di luar rentang umum data Framingham (32–70 tahun). "
                        "Hasil prediksi kurang dapat diandalkan.", "warn")

    patient = dict(male=male, age=age, education=education, cigsPerDay=cigs,
                   BPMeds=int(bpm), prevalentStroke=int(stroke), prevalentHyp=int(hyp),
                   diabetes=int(diab), totChol=chol, sysBP=sys_bp, diaBP=dia_bp,
                   BMI=bmi, heartRate=hr, glucose=glucose)

    with right:
        st.markdown("#### Hasil prediksi")
        if available:
            result = predictor.predict(patient, loaded)
            label, votes, n = predictor.decide(result)
            with card():
                render_result(result, label, votes, n, "Prediksi langsung dari model terlatih.")
            if len(available) < 3:
                missing = [m for m in MODELS if m not in loaded]
                callout(f"Model belum termuat: {', '.join(missing)}. Keputusan memakai model yang tersedia saja.")
            with card():
                st.markdown("**Faktor yang memengaruhi prediksi pasien ini**")
                names = [m for m in ("XGBoost", "EBM") if m in loaded]
                ex_tabs = st.tabs(names or ["-"])
                for tab, m in zip(ex_tabs, names):
                    with tab:
                        try:
                            ex = (predictor.explain_xgb if m == "XGBoost" else predictor.explain_ebm)(patient, loaded)
                            show_plot(contribution_figure(ex.head(10), m))
                        except Exception as e:  # noqa: BLE001
                            st.caption(f"Penjelasan {m} tidak tersedia: {type(e).__name__}")
                st.caption("Merah = mendorong ke arah CHD, hijau = menjauhkan dari CHD (log-odds). "
                           "TabNet dibahas di halaman Interpretabilitas.")
            with st.expander("Detail angka"):
                st.dataframe(
                    result.assign(Probabilitas=(result["Probabilitas"] * 100).round(2),
                                  Threshold=(result["Threshold"] * 100).round(0),
                                  Prediksi=result["Prediksi"].map({1: "CHD", 0: "No CHD"})),
                    hide_index=True, width="stretch")
        else:
            name = st.session_state.get("preset")
            if name not in EXAMPLE_CASES or EXAMPLE_CASES[name]["notebook_prob"] is None:
                if name in EXAMPLE_CASES:
                    callout("Kasus ini belum punya hasil dari notebook. Probabilitasnya baru bisa "
                            "ditampilkan setelah file model dipasang di folder <code>models/</code>. "
                            "Sementara ditampilkan contoh lain dari notebook.", "warn")
                else:
                    callout("File model belum ada di folder <code>models/</code>, jadi prediksi dari input "
                            "belum tersedia. Di bawah ini contoh hasil dari notebook (bukan dari input di kiri).")
                name = FALLBACK_CASE
                st.caption(f"Ditampilkan: {name}.")
            else:
                callout("File model belum ada di folder <code>models/</code>. Ini hasil kasus yang sudah "
                        "dijalankan di notebook (bukan prediksi dari input di kiri).")
            probs = EXAMPLE_CASES[name]["notebook_prob"]
            result = pd.DataFrame({"Model": MODELS,
                                   "Probabilitas": [probs[m] / 100 for m in MODELS],
                                   "Threshold": [THRESHOLDS[m] for m in MODELS]})
            result["Prediksi"] = (result["Probabilitas"] >= result["Threshold"]).astype(int)
            label, votes, n = predictor.decide(result)
            with card():
                render_result(result, label, votes, n, "Angka dari output notebook.")

        with st.expander("Verifikasi vs notebook (cek apakah model sama dengan notebook)"):
            if loaded:
                chk = predictor.verify_against_notebook(loaded)
                if len(chk):
                    st.dataframe(chk, hide_index=True, width="stretch")
                    if chk["Cocok"].all():
                        st.success("Semua cocok dengan output notebook.")
                    else:
                        st.error("Ada selisih: file model/scaler/versi library tidak sama dengan notebook.")
                st.json(predictor.env_info(loaded))
            else:
                st.caption("Model belum termuat.")

        with st.expander("Status model"):
            for m in MODELS:
                if m in loaded:
                    st.markdown(f"✅ **{m}** termuat")
                else:
                    st.markdown(f"⚪ **{m}** belum termuat: `{errors.get(m, 'tidak ada informasi')}`")

    callout("Alat ini adalah prototipe penelitian untuk tujuan akademik. Bukan alat diagnosis dan tidak "
            "menggantikan penilaian tenaga medis.", "warn", "⚕️")



# ------------------------------------------------------------------ simulasi smartwatch
SW_DEFAULTS = dict(male=0, age=55, education=2, cigsPerDay=0, BPMeds=False, prevalentStroke=False,
                   prevalentHyp=False, diabetes=False, totChol=220, BMI=26.0, glucose=90,
                   heartRate=75, sysBP=130, diaBP=82)
SW_VITALS = ("heartRate", "sysBP", "diaBP")


def apply_sw_preset() -> None:
    name = st.session_state.get("sw_preset")
    if name in EXAMPLE_CASES:
        for k, v in EXAMPLE_CASES[name]["inputs"].items():
            st.session_state[f"sw_{k}"] = bool(v) if k in BINARY else v


def simulate_prediction(patient: dict, loaded: dict):
    """Returns (result_df | None, label, votes, n, source)."""
    if any(m in loaded for m in MODELS):
        res = predictor.predict(patient, loaded)
        label, votes, n = predictor.decide(res)
        return res, label, votes, n, "live"
    case = EXAMPLE_CASES.get(st.session_state.get("sw_preset"))
    if case and case["notebook_prob"] and all(float(patient[k]) == float(case["inputs"][k]) for k in FEATURES):
        probs = case["notebook_prob"]
        res = pd.DataFrame({"Model": MODELS, "Probabilitas": [probs[m] / 100 for m in MODELS],
                            "Threshold": [THRESHOLDS[m] for m in MODELS]})
        res["Prediksi"] = (res["Probabilitas"] >= res["Threshold"]).astype(int)
        label, votes, n = predictor.decide(res)
        return res, label, votes, n, "notebook"
    return None, None, None, None, "none"


def page_watch() -> None:
    hero("Simulasi Smartwatch",
         "Ketuk tombol di jam, tunggu pengukuran detak jantung dan tekanan darah, lalu lihat "
         "prediksi risiko CHD langsung di layar jam.",
         tag="Simulasi · Bukan alat diagnosis")
    loaded, _ = get_models()
    for k, v in SW_DEFAULTS.items():
        st.session_state.setdefault(f"sw_{k}", v)

    callout("<b>Simulasi:</b> nilai detak jantung, sistolik, dan diastolik di sini diatur lewat slider, bukan "
            "dibaca dari sensor sungguhan. Di dunia nyata, kebanyakan smartwatch bisa mengukur detak jantung, "
            "tetapi umumnya belum mengukur tekanan darah secara akurat tanpa kalibrasi dengan manset.")

    watch_col, ctrl_col = st.columns([4, 6], gap="large")

    # ---------------- kontrol
    with ctrl_col:
        with card():
            st.markdown("#### 🎛️ Pengaturan simulasi")
            st.selectbox("Pilih pengguna contoh", ["Atur sendiri"] + list(EXAMPLE_CASES), key="sw_preset",
                         on_change=apply_sw_preset,
                         help="Memilih pengguna contoh akan mengisi profil dan hasil pengukuran sekaligus.")
            st.markdown("**⌚ Nilai yang akan terbaca oleh jam**")
            st.slider("Detak jantung (BPM)", 40, 150, key="sw_heartRate")
            a, b = st.columns(2)
            a.slider("Sistolik (mmHg)", 80, 220, key="sw_sysBP")
            b.slider("Diastolik (mmHg)", 40, 130, key="sw_diaBP")
            if st.session_state["sw_sysBP"] <= st.session_state["sw_diaBP"]:
                callout("Sistolik seharusnya lebih tinggi dari diastolik.", "warn")
            st.checkbox("Mode cepat (animasi lebih singkat)", key="sw_fast")

            with st.expander("📱 Profil pengguna (data dari aplikasi kesehatan di ponsel)"):
                st.caption("Jam hanya mengukur tiga nilai di atas. Model membutuhkan 11 data lain, "
                           "yang di simulasi ini dianggap tersimpan di profil pengguna.")
                a, b = st.columns(2)
                a.selectbox("Jenis kelamin", [0, 1], key="sw_male", format_func=lambda v: "Pria" if v else "Wanita")
                b.number_input("Usia (tahun)", 20, 90, step=1, key="sw_age")
                a.selectbox("Pendidikan", [1, 2, 3, 4], key="sw_education", format_func=lambda v: EDU[v])
                b.number_input("Rokok per hari", 0, 70, step=1, key="sw_cigsPerDay")
                a.number_input("Kolesterol total (mg/dL)", 100, 700, step=1, key="sw_totChol")
                b.number_input("Glukosa (mg/dL)", 40, 400, step=1, key="sw_glucose")
                a.number_input("BMI", 14.0, 60.0, step=0.1, key="sw_BMI")
                t1, t2 = st.columns(2)
                t1.toggle("Riwayat hipertensi", key="sw_prevalentHyp")
                t1.toggle("Obat tekanan darah", key="sw_BPMeds")
                t2.toggle("Diabetes", key="sw_diabetes")
                t2.toggle("Riwayat stroke", key="sw_prevalentStroke")

    s = st.session_state
    patient = dict(male=s["sw_male"], age=s["sw_age"], education=s["sw_education"],
                   cigsPerDay=s["sw_cigsPerDay"], BPMeds=int(s["sw_BPMeds"]),
                   prevalentStroke=int(s["sw_prevalentStroke"]), prevalentHyp=int(s["sw_prevalentHyp"]),
                   diabetes=int(s["sw_diabetes"]), totChol=s["sw_totChol"], sysBP=s["sw_sysBP"],
                   diaBP=s["sw_diaBP"], BMI=s["sw_BMI"], heartRate=s["sw_heartRate"], glucose=s["sw_glucose"])
    result, label, votes, n, source = simulate_prediction(patient, loaded)

    payload = {
        "vitals": {"hr": int(patient["heartRate"]), "sys": int(patient["sysBP"]), "dia": int(patient["diaBP"])},
        "models": None if result is None else [
            {"name": r.Model, "prob": float(r.Probabilitas), "thr": float(r.Threshold),
             "pred": int(r.Prediksi), "color": MODEL_COLORS[r.Model]} for r in result.itertuples()],
        "label": label, "votes": votes, "n": n, "source": source,
        "speed": 0.35 if s.get("sw_fast") else 1.0,
    }

    # ---------------- jam
    with watch_col:
        with card():
            render_html(watch_html(payload), 540)
            st.caption("Ketuk tombol **Cek kesehatan** di layar jam. Mengubah slider atau profil akan "
                       "mengembalikan jam ke layar awal.")

    # ---------------- di balik layar
    with ctrl_col:
        with card():
            st.markdown("#### 🧠 Di balik layar")
            st.markdown(
                '<div class="pipe"><span class="b">⌚ Sensor jam<br>3 nilai</span><span class="a">+</span>'
                '<span class="b">📱 Profil ponsel<br>11 data</span><span class="a">→</span>'
                '<span class="b">XGBoost · EBM · TabNet</span><span class="a">→</span>'
                '<span class="b">Majority voting<br>≥ 2 dari 3</span><span class="a">→</span>'
                '<span class="b">⌚ Layar jam</span></div>',
                unsafe_allow_html=True)
            with st.expander("Lihat 14 fitur yang dikirim ke model"):
                st.dataframe(
                    pd.DataFrame({
                        "Fitur": FEATURES,
                        "Nilai": [str(patient[f]) for f in FEATURES],
                        "Sumber": ["⌚ Jam" if f in SW_VITALS else "📱 Profil" for f in FEATURES],
                    }), hide_index=True, width="stretch")
            if result is not None:
                with st.expander("Detail prediksi (menampilkan hasil lebih dulu)"):
                    src = "Prediksi langsung dari model terlatih." if source == "live" else "Angka dari output notebook."
                    render_result(result, label, votes, n, src)
            else:
                callout("File model belum ada di folder <code>models/</code>, dan kombinasi nilai ini bukan "
                        "salah satu kasus yang sudah dijalankan di notebook, jadi jam akan menampilkan "
                        "\"Hasil belum tersedia\". Pilih salah satu pengguna contoh tanpa mengubah nilainya, "
                        "atau pasang file model.", "warn")

    callout("Hasil adalah estimasi risiko CHD 10 tahun dari model penelitian, bukan diagnosis. Keputusan "
            "medis tetap perlu pemeriksaan oleh tenaga kesehatan.", "warn", "⚕️")


# ------------------------------------------------------------------ metodologi
def page_method() -> None:
    hero("Metodologi", "Dari data mentah Framingham sampai model siap dievaluasi.",
         tag="Alur penelitian")
    d = DATASET

    section("Perjalanan data")
    c = st.columns(4, gap="medium")
    stats = [(f"{d['sebelum_drop']:,}", "pasien awal"),
             (f"{d['setelah_drop']:,}", f"setelah hapus missing ({d['dihapus']} baris)"),
             (f"{sum(d['kelas_setelah_balancing'].values()):,}", "setelah penyeimbangan kelas"),
             ("70 / 15 / 15", "train / validation / test")]
    for col, (v, l) in zip(c, stats):
        with col:
            with card():
                st.markdown(f'<div class="stat"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)

    section("Langkah pra-pemrosesan")
    steps = [
        f"<b>Dataset:</b> {d['sumber']}, {d['sebelum_drop']:,} pasien.",
        f"<b>Missing value:</b> baris kosong dihapus ({d['dihapus']} baris, 13,73%). Tersisa "
        f"{d['setelah_drop']:,} pasien: {d['kelas_setelah_drop']['No CHD']:,} tanpa CHD dan "
        f"{d['kelas_setelah_drop']['CHD']} CHD.",
        "<b>Fitur:</b> kolom <code>currentSmoker</code> dibuang; 14 fitur dipakai.",
        f"<b>Penyeimbangan kelas:</b> SMOTE (rasio 0,8) lalu RandomUnderSampler (rasio 0,8), hasilnya "
        f"{d['kelas_setelah_balancing']['No CHD']:,} tanpa CHD dan {d['kelas_setelah_balancing']['CHD']:,} CHD.",
        f"<b>Pembagian data:</b> dilakukan setelah penyeimbangan, stratified 70/15/15 (random_state 42): "
        f"train {d['split']['Train']:,}, validation {d['split']['Validation']}, test {d['split']['Test']}. "
        f"Validation dan test masing-masing {d['kelas_test']['No CHD']} tanpa CHD dan {d['kelas_test']['CHD']} CHD. "
        "Karena SMOTE dijalankan sebelum pembagian, validation dan test juga memuat sampel sintetis.",
        "<b>Skala:</b> StandardScaler (fit pada train saja) hanya untuk TabNet; XGBoost dan EBM memakai data mentah.",
        "<b>Tuning hyperparameter:</b> XGBoost GridSearchCV 3.888 kandidat dan EBM GridSearchCV 216 kandidat "
        "(keduanya 5-fold stratified, skor ROC-AUC); TabNet grid search manual 36 kombinasi dengan Validation ROC-AUC.",
        "<b>Threshold:</b> dicari di validation set dengan memaksimalkan F2 (rentang 0,20 sampai 0,99, langkah 0,01; "
        "jika F2 sama dipilih threshold lebih tinggi), lalu dikunci: XGBoost 0,22 · EBM 0,20 · TabNet 0,23.",
        "<b>Evaluasi:</b> test set dengan threshold terkunci; recall sebagai metrik utama, dilengkapi F2, precision, "
        "ROC-AUC, accuracy, dan F1.",
        "<b>Keputusan akhir:</b> majority voting, risiko tinggi jika minimal 2 dari 3 model memprediksi CHD.",
        "<b>Interpretabilitas:</b> SHAP dan LIME (XGBoost, TabNet) serta global dan local explanation EBM.",
    ]
    with card():
        st.markdown("".join(f'<div class="step"><div class="n">{i}</div><div class="s">{s}</div></div>'
                            for i, s in enumerate(steps, 1)), unsafe_allow_html=True)

    section("Distribusi kelas")
    with card():
        show_image("smote_balance.png", "Distribusi kelas sebelum dan sesudah penyeimbangan")

    section("Fitur yang dipakai")
    with card():
        st.dataframe(pd.DataFrame({"Fitur": FEATURES, "Keterangan": [FEATURE_INFO[f] for f in FEATURES]}),
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ navigasi
PAGES = {
    "🏠  Ringkasan": page_summary,
    "📊  Perbandingan model": page_compare,
    "🧩  Detail per model": page_detail,
    "🔍  Interpretabilitas": page_xai,
    "🩺  Prediksi pasien": page_predict,
    "⌚  Simulasi smartwatch": page_watch,
    "📋  Metodologi": page_method,
}

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-logo">{LOGO}</div><div><div class="brand-name">CHD Risk</div>'
        f'<div class="brand-sub">XGBoost · EBM · TabNet</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    choice = st.radio("Halaman", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Prototipe penelitian untuk tujuan akademik. Bukan alat diagnosis.")

PAGES[choice]()
