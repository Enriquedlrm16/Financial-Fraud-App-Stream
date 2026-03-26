#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard de detección de fraude en tiempo real - V3 optimizada
Ejecución:
    streamlit run fraud_realtime_dashboard_v3_optimized.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DEFAULT_FILE_PATH = Path("PS_20174392719_1491204439457_log.csv")

REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]

MODEL_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "dest_is_merchant", "dest_is_customer", "high_value", "empties_account",
    "balance_error_orig", "balance_error_dest", "type_encoded",
]

PALETTE = {
    "bg": "#0b1020",
    "line": "rgba(255,255,255,0.08)",
    "text": "#f5f7fb",
    "muted": "#99a4bf",
    "cyan": "#00d9ff",
    "green": "#00ff88",
    "red": "#ff5c7c",
    "yellow": "#ffd166",
    "purple": "#b388ff",
    "blue": "#4dabff",
    "orange": "#ff9f43",
}

FAST_TRAIN_ROWS_DEFAULT = 250_000
VIS_SAMPLE_SIZE_DEFAULT = 25_000


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(circle at top left, #111a33 0%, {PALETTE["bg"]} 45%, #060a15 100%);
            color: {PALETTE["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #11172b 0%, #0c1222 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }}
        h1, h2, h3, h4 {{
            color: {PALETTE["text"]};
            letter-spacing: -0.02em;
        }}
        .hero {{
            padding: 1.35rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(0,217,255,0.12) 0%, rgba(179,136,255,0.09) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 14px 34px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }}
        .hero-title {{
            font-size: 2.55rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            line-height: 1.05;
        }}
        .hero-sub {{
            color: {PALETTE["muted"]};
            font-size: 1rem;
        }}
        .section-card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1rem 1rem 0.5rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .kpi-card {{
            border-radius: 18px;
            padding: 16px 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 104px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.20);
        }}
        .kpi-label {{
            color: {PALETTE["muted"]};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 2rem;
            line-height: 1;
            font-weight: 800;
            color: {PALETTE["text"]};
            margin-bottom: 0.35rem;
        }}
        .kpi-hint {{
            color: {PALETTE["muted"]};
            font-size: 0.82rem;
        }}
        .accent-cyan .kpi-value {{ color: {PALETTE["cyan"]}; }}
        .accent-green .kpi-value {{ color: {PALETTE["green"]}; }}
        .accent-red .kpi-value {{ color: {PALETTE["red"]}; }}
        .accent-yellow .kpi-value {{ color: {PALETTE["yellow"]}; }}
        .accent-purple .kpi-value {{ color: {PALETTE["purple"]}; }}
        .accent-blue .kpi-value {{ color: {PALETTE["blue"]}; }}
        .pill {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 6px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .pill.info {{ background: rgba(0,217,255,0.12); color: {PALETTE["cyan"]}; }}
        .pill.good {{ background: rgba(0,255,136,0.12); color: {PALETTE["green"]}; }}
        .pill.warn {{ background: rgba(255,209,102,0.12); color: {PALETTE["yellow"]}; }}
        .pill.danger {{ background: rgba(255,92,124,0.12); color: {PALETTE["red"]}; }}
        .risk-grid {{
            display:grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 12px;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }}
        .risk-card {{
            border-radius: 16px;
            padding: 14px 16px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
        }}
        .risk-title {{
            font-size: 0.8rem;
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }}
        .risk-value {{
            font-size: 1.45rem;
            font-weight: 800;
        }}
        .upload-zone {{
            border-radius: 18px;
            padding: 18px 18px;
            background: linear-gradient(135deg, rgba(0,217,255,0.08), rgba(0,255,136,0.05));
            border: 1px dashed rgba(0,217,255,0.35);
            margin-bottom: 1rem;
        }}
        .status-box {{
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }}
        .small-muted {{
            color: {PALETTE["muted"]};
            font-size: 0.86rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            background: transparent;
            color: {PALETTE["text"]};
            font-weight: 700;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 2px solid {PALETTE["cyan"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_theme(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color=PALETTE["text"]),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor=PALETTE["line"], zeroline=False, linecolor=PALETTE["line"]),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["line"], zeroline=False, linecolor=PALETTE["line"]),
    )
    return fig


def ensure_required_columns(df: pd.DataFrame, require_target: bool = True) -> None:
    needed = REQUIRED_COLUMNS if require_target else [c for c in REQUIRED_COLUMNS if c != "isFraud"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")


def build_features(df: pd.DataFrame, amount_threshold: float, encoder: LabelEncoder | None = None):
    data = df.copy()
    data["balance_diff_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["balance_error_orig"] = data["amount"] - data["balance_diff_orig"]
    data["balance_error_dest"] = data["amount"] - data["balance_diff_dest"]
    data["dest_is_merchant"] = data["nameDest"].astype(str).str.startswith("M").astype(int)
    data["dest_is_customer"] = data["nameDest"].astype(str).str.startswith("C").astype(int)
    data["high_value"] = (data["amount"] > amount_threshold).astype(int)
    data["empties_account"] = (data["newbalanceOrig"] == 0).astype(int)

    if encoder is None:
        encoder = LabelEncoder()
        data["type_encoded"] = encoder.fit_transform(data["type"].astype(str))
    else:
        known = set(encoder.classes_)
        fallback = encoder.classes_[0]
        safe_type = data["type"].astype(str).where(data["type"].astype(str).isin(known), fallback)
        data["type_encoded"] = encoder.transform(safe_type)

    X = data[MODEL_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, encoder


@st.cache_data(show_spinner=False)
def load_training_data(file_path: str, nrows: int) -> pd.DataFrame:
    df = pd.read_csv(file_path, nrows=nrows)
    ensure_required_columns(df, require_target=True)
    return df


@st.cache_data(show_spinner=False)
def sample_training_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    fraud_df = df[df["isFraud"] == 1]
    normal_df = df[df["isFraud"] == 0]
    n_fraud = min(len(fraud_df), max(1, int(n * 0.25)))
    n_normal = min(len(normal_df), max(0, n - n_fraud))
    parts = []
    if n_fraud > 0:
        parts.append(fraud_df.sample(n=n_fraud, random_state=42))
    if n_normal > 0:
        parts.append(normal_df.sample(n=n_normal, random_state=42))
    out = pd.concat(parts, ignore_index=True) if parts else df.sample(n=n, random_state=42)
    return out.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def precompute_train_step_series(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("step")
        .agg(total=("step", "size"), fraudes=("isFraud", "sum"))
        .reset_index()
        .sort_values("step")
    )


@st.cache_resource(show_spinner=False)
def train_model(file_path: str, nrows: int):
    df = load_training_data(file_path, nrows=nrows)
    amount_threshold = float(df["amount"].quantile(0.95))
    X, encoder = build_features(df, amount_threshold=amount_threshold, encoder=None)
    y = df["isFraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    feature_importance = (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    training_scored = df.copy()

    return {
        "df": df,
        "model": model,
        "encoder": encoder,
        "amount_threshold": amount_threshold,
        "metrics": {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "feature_importance": feature_importance,
        "training_scored": training_scored,
    }


def score_new_data(df_new: pd.DataFrame, artifacts: dict, decision_threshold: float = 0.20) -> pd.DataFrame:
    ensure_required_columns(df_new, require_target=("isFraud" in df_new.columns))
    X_new, _ = build_features(df_new, amount_threshold=artifacts["amount_threshold"], encoder=artifacts["encoder"])
    scored = df_new.copy()
    scored["fraud_probability"] = artifacts["model"].predict_proba(X_new)[:, 1]
    scored["fraud_pred_model"] = artifacts["model"].predict(X_new)
    scored["fraud_pred"] = (scored["fraud_probability"] >= decision_threshold).astype(int)
    scored["risk_level"] = pd.cut(
        scored["fraud_probability"],
        bins=[-0.01, 0.05, 0.20, 0.50, 1.00],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
    )
    return scored


def format_compact_currency(value: float) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def render_kpi_cards(kpis):
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card accent-{kpi.get('accent','cyan')}">
                    <div class="kpi-label">{kpi['label']}</div>
                    <div class="kpi-value">{kpi['value']}</div>
                    <div class="kpi-hint">{kpi.get('hint','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def filter_panel(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Filtros")

    c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.1, 1.2])

    with c1:
        types = sorted(df["type"].astype(str).dropna().unique().tolist())
        selected_types = st.multiselect("Tipo de transacción", types, default=types, key=f"{mode}_types")
    with c2:
        min_amount = float(df["amount"].min()) if len(df) else 0.0
        max_amount = float(df["amount"].max()) if len(df) else 1.0
        amount_max = max_amount if max_amount > min_amount else min_amount + 1
        amount_range = st.slider("Rango de monto", min_value=float(min_amount), max_value=float(amount_max), value=(float(min_amount), float(amount_max)), key=f"{mode}_amount")
    with c3:
        min_step = int(df["step"].min()) if len(df) else 0
        max_step = int(df["step"].max()) if len(df) else 1
        step_max = max_step if max_step > min_step else min_step + 1
        step_range = st.slider("Rango de step", min_value=min_step, max_value=step_max, value=(min_step, step_max), key=f"{mode}_step")
    with c4:
        min_bal = float(df["oldbalanceOrg"].min()) if len(df) else 0.0
        max_bal = float(df["oldbalanceOrg"].max()) if len(df) else 1.0
        bal_max = max_bal if max_bal > min_bal else min_bal + 1
        balance_range = st.slider("Balance origen", min_value=float(min_bal), max_value=float(bal_max), value=(float(min_bal), float(bal_max)), key=f"{mode}_balance")

    if mode == "pred":
        s1, s2 = st.columns([1, 1.2])
        with s1:
            status = st.radio("Estado predicción", ["Todos", "Solo predichos fraude", "Solo predichos normal"], horizontal=True, key="pred_status")
        with s2:
            prob_range = st.slider("Probabilidad de fraude", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.01, key="pred_prob")
    else:
        s1, _ = st.columns([1, 1])
        with s1:
            status = st.radio("Estado real", ["Todos", "Solo fraude", "Solo normal"], horizontal=True, key="train_status")
        prob_range = None

    out = df.copy()
    out = out[out["type"].isin(selected_types)]
    out = out[out["amount"].between(amount_range[0], amount_range[1])]
    out = out[out["step"].between(step_range[0], step_range[1])]
    out = out[out["oldbalanceOrg"].between(balance_range[0], balance_range[1])]

    if mode == "pred" and "fraud_probability" in out.columns:
        out = out[out["fraud_probability"].between(prob_range[0], prob_range[1])]
        if status == "Solo predichos fraude":
            out = out[out["fraud_pred"] == 1]
        elif status == "Solo predichos normal":
            out = out[out["fraud_pred"] == 0]
    else:
        if status == "Solo fraude":
            out = out[out["isFraud"] == 1]
        elif status == "Solo normal":
            out = out[out["isFraud"] == 0]

    st.markdown('</div>', unsafe_allow_html=True)
    return out


def fig_type_distribution(df: pd.DataFrame, fraud_col: str):
    grouped = df.groupby("type").agg(total=("type", "size"), fraudes=(fraud_col, "sum")).reset_index().sort_values("total", ascending=False)
    grouped["normales"] = grouped["total"] - grouped["fraudes"]
    fig = go.Figure()
    fig.add_bar(name="Normal", x=grouped["type"], y=grouped["normales"], marker_color="rgba(77,171,255,0.75)")
    fig.add_bar(name="Fraude", x=grouped["type"], y=grouped["fraudes"], marker_color="rgba(255,92,124,0.9)")
    fig.update_layout(title="Fraude vs normal por tipo", barmode="stack")
    return plotly_theme(fig, height=430)


def fig_time_series(df: pd.DataFrame, fraud_col: str | None = None):
    if {"step", "total", "fraudes"}.issubset(df.columns):
        grouped = df.sort_values("step").copy()
    else:
        grouped = df.groupby("step").agg(total=("step", "size"), fraudes=(fraud_col, "sum")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=grouped["step"], y=grouped["total"], mode="lines", name="Total", line=dict(color=PALETTE["cyan"], width=2.5))
    fig.add_scatter(x=grouped["step"], y=grouped["fraudes"], mode="lines", name="Fraudes", line=dict(color=PALETTE["red"], width=2.5))
    fig.update_layout(title="Serie temporal por step", xaxis_title="Step", yaxis_title="Cantidad")
    return plotly_theme(fig, height=430)


def fig_histogram(df: pd.DataFrame):
    fig = px.histogram(df, x="amount", nbins=35, title="Distribución de montos", opacity=0.85)
    fig.update_traces(marker_color=PALETTE["blue"], marker_line_width=0)
    fig.update_layout(xaxis_title="Monto", yaxis_title="Frecuencia")
    return plotly_theme(fig, height=430)


def fig_scatter(df: pd.DataFrame, fraud_col: str):
    sample = df.sample(min(4000, len(df)), random_state=42) if len(df) > 4000 else df.copy()
    sample["estado"] = np.where(sample[fraud_col] == 1, "Fraude", "Normal")
    fig = px.scatter(sample, x="oldbalanceOrg", y="amount", color="estado", color_discrete_map={"Normal": PALETTE["cyan"], "Fraude": PALETTE["red"]}, title="Dispersión: monto vs balance origen", hover_data=["type"], opacity=0.75)
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(xaxis_title="Balance origen", yaxis_title="Monto")
    return plotly_theme(fig, height=480)


def fig_heatmap(df: pd.DataFrame, fraud_col: str):
    corr_df = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].copy()
    corr_df[fraud_col] = df[fraud_col].astype(float)
    corr = corr_df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Mapa de calor de correlaciones", color_continuous_scale=[[0.0, "#243b6b"], [0.5, "#0f1728"], [1.0, "#00d9ff"]])
    fig.update_layout(coloraxis_colorbar=dict(title="Corr"))
    return plotly_theme(fig, height=480)


def fig_roc(artifacts: dict):
    fpr = artifacts["roc_curve"]["fpr"]
    tpr = artifacts["roc_curve"]["tpr"]
    auc = artifacts["metrics"]["roc_auc"]
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo (AUC={auc:.4f})", line=dict(color=PALETTE["green"], width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(color=PALETTE["muted"], width=2, dash="dash"))
    fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
    return plotly_theme(fig, height=430)


def fig_feature_importance(artifacts: dict):
    fi = artifacts["feature_importance"].sort_values("importance", ascending=True)
    fig = px.bar(fi, x="importance", y="feature", orientation="h", title="Importancia de variables")
    fig.update_traces(marker_color=PALETTE["purple"])
    fig.update_layout(xaxis_title="Importancia", yaxis_title="")
    return plotly_theme(fig, height=460)


def render_risk_summary(df: pd.DataFrame) -> None:
    if "risk_level" not in df.columns:
        return
    counts = df["risk_level"].value_counts().to_dict()
    cards = [("Bajo", counts.get("Bajo", 0), PALETTE["green"]), ("Medio", counts.get("Medio", 0), PALETTE["yellow"]), ("Alto", counts.get("Alto", 0), PALETTE["orange"]), ("Crítico", counts.get("Crítico", 0), PALETTE["red"])]
    html = ['<div class="risk-grid">']
    for title, value, color in cards:
        html.append(f'<div class="risk-card"><div class="risk-title">{title}</div><div class="risk-value" style="color:{color}">{int(value):,}</div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def top_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "fraud_probability", "fraud_pred", "fraud_pred_model", "risk_level", "isFraud"] if c in df.columns]
    table = df.sort_values("fraud_probability", ascending=False)[cols].head(50).copy()
    if "fraud_probability" in table.columns:
        table["fraud_probability"] = table["fraud_probability"].round(4)
    return table


st.set_page_config(page_title="Fraud Dashboard Pro", layout="wide")
inject_css()

with st.sidebar:
    st.markdown("## Configuración")
    training_path = st.text_input("Ruta CSV entrenamiento", value=str(DEFAULT_FILE_PATH))
    fast_train_rows = st.number_input("Filas para entrenamiento rápido", min_value=50_000, max_value=1_000_000, value=FAST_TRAIN_ROWS_DEFAULT, step=50_000)
    vis_sample_size = st.number_input("Filas para visualización", min_value=5_000, max_value=100_000, value=VIS_SAMPLE_SIZE_DEFAULT, step=5_000)
    pred_threshold = st.slider("Umbral de decisión", min_value=0.01, max_value=0.90, value=0.20, step=0.01)
    st.markdown(
        '<div class="small-muted">La carga inicial usa un subconjunto controlado del dataset de entrenamiento para reducir el tiempo de arranque. El umbral de decisión permite adaptar la sensibilidad del scoring en los CSV de prueba.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hero"><div class="hero-title">Dashboard de detección de fraude en tiempo real</div><div class="hero-sub">Monitoreo visual, priorización de riesgo y scoring predictivo sobre nuevos archivos CSV.</div></div>',
    unsafe_allow_html=True,
)

with st.spinner("Inicializando modelo, cargando subconjunto de entrenamiento y preparando la vista inicial..."):
    try:
        artifacts = train_model(training_path, int(fast_train_rows))
    except Exception as exc:
        st.error(f"No pude cargar o entrenar con el dataset base: {exc}")
        st.stop()

training_df = artifacts["training_scored"]
training_vis_df = sample_training_df(training_df, int(vis_sample_size))
train_step_series_full = precompute_train_step_series(training_df)
metrics = artifacts["metrics"]

st.markdown(
    f'<div class="status-box"><strong>Estado de carga</strong><br>'
    f'Se ha inicializado la aplicación con <strong>{len(training_df):,}</strong> filas de entrenamiento y '
    f'<strong>{len(training_vis_df):,}</strong> filas para visualización interactiva. '
    f'Esta configuración prioriza una carga inicial más rápida manteniendo la predicción en tiempo real sobre los CSV que subas.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<span class="pill info">Entrenamiento cargado: {len(training_df):,} filas</span>'
    f'<span class="pill good">ROC-AUC {metrics["roc_auc"]:.4f}</span>'
    f'<span class="pill warn">Umbral alto valor {format_compact_currency(artifacts["amount_threshold"])}</span>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Modelo + entrenamiento", "Predicción en CSV nuevo"])

with tab1:
    render_kpi_cards([
        {"label": "Transacciones", "value": f"{len(training_df):,}", "hint": "Subconjunto cargado en memoria", "accent": "cyan"},
        {"label": "Fraudes", "value": f"{int(training_df['isFraud'].sum()):,}", "hint": f"{training_df['isFraud'].mean()*100:.2f}% del total cargado", "accent": "red"},
        {"label": "Monto total", "value": format_compact_currency(training_df['amount'].sum()), "hint": "Volumen analizado", "accent": "blue"},
        {"label": "Monto fraude", "value": format_compact_currency(training_df.loc[training_df['isFraud'] == 1, 'amount'].sum()), "hint": "Fraude observado", "accent": "yellow"},
        {"label": "Precisión", "value": f"{metrics['precision']:.4f}", "hint": "Conjunto de prueba interno", "accent": "green"},
        {"label": "Recall", "value": f"{metrics['recall']:.4f}", "hint": "Sensibilidad del modelo", "accent": "purple"},
    ])

    train_filtered = filter_panel(training_vis_df, mode="train")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_type_distribution(train_filtered, "isFraud"), use_container_width=True)
    with c2:
        st.plotly_chart(fig_histogram(train_filtered), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(fig_heatmap(train_filtered, "isFraud"), use_container_width=True)
    with d2:
        st.plotly_chart(fig_scatter(train_filtered, "isFraud"), use_container_width=True)

    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(fig_time_series(train_step_series_full), use_container_width=True)
    with e2:
        st.plotly_chart(fig_roc(artifacts), use_container_width=True)

    st.plotly_chart(fig_feature_importance(artifacts), use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top transacciones observadas en el entrenamiento")
    train_top = (
        training_df[training_df["isFraud"] == 1]
        .sort_values(["amount", "step"], ascending=[False, True])[["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]]
        .head(50)
        .copy()
    )
    st.dataframe(train_top, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="upload-zone"><strong>Carga un CSV para análisis</strong><br>El modelo calculará <code>fraud_pred</code>, <code>fraud_probability</code> y <code>risk_level</code> sobre cada transacción del archivo.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV para scoring", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.info("Esperando archivo de entrada para ejecutar el scoring en tiempo real.")
    else:
        try:
            df_new = pd.read_csv(uploaded)
            ensure_required_columns(df_new, require_target=("isFraud" in df_new.columns))
            scored_df = score_new_data(df_new, artifacts, decision_threshold=float(pred_threshold))
        except Exception as exc:
            st.error(f"No pude procesar el CSV nuevo: {exc}")
            st.stop()

        st.markdown(
            f'<span class="pill info">Umbral de decisión activo: {pred_threshold:.2f}</span>'
            f'<span class="pill warn">Predicción manual por probabilidad</span>',
            unsafe_allow_html=True,
        )

        render_kpi_cards([
            {"label": "Transacciones", "value": f"{len(scored_df):,}", "hint": "Archivo cargado", "accent": "cyan"},
            {"label": "Predichas fraude", "value": f"{int(scored_df['fraud_pred'].sum()):,}", "hint": f"{scored_df['fraud_pred'].mean()*100:.2f}% con umbral actual", "accent": "red"},
            {"label": "Probabilidad media", "value": f"{scored_df['fraud_probability'].mean():.2%}", "hint": "Riesgo promedio", "accent": "yellow"},
            {"label": "Monto total", "value": format_compact_currency(scored_df['amount'].sum()), "hint": "Volumen del CSV", "accent": "blue"},
            {"label": "Monto con fraude pred.", "value": format_compact_currency(scored_df.loc[scored_df['fraud_pred'] == 1, 'amount'].sum()), "hint": "Monto priorizado", "accent": "green"},
            {"label": "Máxima prob.", "value": f"{scored_df['fraud_probability'].max():.2%}", "hint": "Pico de riesgo", "accent": "purple"},
        ])

        render_risk_summary(scored_df)

        if "isFraud" in scored_df.columns:
            eval_precision = precision_score(scored_df["isFraud"], scored_df["fraud_pred"], zero_division=0)
            eval_recall = recall_score(scored_df["isFraud"], scored_df["fraud_pred"], zero_division=0)
            eval_f1 = f1_score(scored_df["isFraud"], scored_df["fraud_pred"], zero_division=0)
            st.markdown(
                f'<span class="pill info">Precisión del archivo: {eval_precision:.4f}</span>'
                f'<span class="pill good">Recall del archivo: {eval_recall:.4f}</span>'
                f'<span class="pill warn">F1 del archivo: {eval_f1:.4f}</span>',
                unsafe_allow_html=True,
            )

        pred_filtered = filter_panel(scored_df, mode="pred")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_type_distribution(pred_filtered, "fraud_pred"), use_container_width=True)
        with c2:
            st.plotly_chart(fig_histogram(pred_filtered), use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(fig_heatmap(pred_filtered, "fraud_pred"), use_container_width=True)
        with d2:
            st.plotly_chart(fig_scatter(pred_filtered, "fraud_pred"), use_container_width=True)

        st.plotly_chart(fig_time_series(pred_filtered, "fraud_pred"), use_container_width=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Top 50 transacciones con mayor riesgo")
        top_risk = top_risk_table(pred_filtered)
        st.dataframe(top_risk, use_container_width=True, hide_index=True)

        csv_bytes = pred_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV con predicciones", data=csv_bytes, file_name="predicciones_fraude.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)
