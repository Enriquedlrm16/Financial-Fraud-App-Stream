# 🛡️ Fraud Detection Dashboard — Real-Time Scoring

An interactive **Streamlit dashboard** for real-time financial transaction fraud detection. Trains a Random Forest model on the PaySim dataset, exposes full model diagnostics, and scores any new CSV on the fly — returning fraud probability, risk level and a downloadable results file.

> 🚀 **Live app:** [your-streamlit-url-here]

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│              TRAINING DATASET (PaySim)                       │
│         PS_20174392719_1491204439457_log.csv                 │
│              loaded via Git LFS (6M+ rows)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │  nrows configurable (default 250k)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                          │
│                                                              │
│  balance_error_orig    balance_error_dest                    │
│  dest_is_merchant      dest_is_customer                      │
│  high_value (>p95)     empties_account                       │
│  type_encoded (LabelEncoder)                                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           Random Forest (class_weight=balanced)              │
│      100 trees · max_depth=15 · stratified split 80/20       │
│      Metrics: Precision · Recall · F1 · ROC-AUC              │
└─────────────┬────────────────────────────┬───────────────────┘
              │                            │
              ▼                            ▼
   TAB 1 — Model + Training        TAB 2 — Score new CSV
   ─────────────────────────       ──────────────────────────
   KPI cards                       Upload any PaySim CSV
   Type distribution               fraud_probability per row
   Amount histogram                risk_level classification
   Heatmap + scatter               KPI cards + charts
   Time series (by step)           Top 50 high-risk table
   ROC curve                       Download scored CSV
   Feature importance
   Top fraud transactions
```

---

## 🗂️ Repository Structure

```
fraud-detection-dashboard/
│
├── fraud_realtime_dashboard_v4_demo_scoring.py   # Main Streamlit app
│
├── paysim_sample_500.csv          # 500-row sample for quick testing
├── paysim_synthetic_1000.csv      # 1000 synthetic rows (balanced fraud)
│
├── PS_20174392719_1491204439457_log.csv  # Full PaySim dataset — Git LFS
│
├── requirements.txt
├── .gitattributes                 # Git LFS tracking config
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Training (Tab 1)

On startup the app loads up to **250,000 rows** of the PaySim dataset (configurable in the sidebar), engineers features and trains a Random Forest with `class_weight="balanced"` to handle the heavily imbalanced fraud labels.

The sampling strategy ensures fraud cases are always well-represented: the internal sampler targets **25% fraud / 75% normal** from the available data before training.

Results are cached with `@st.cache_resource` so the model is only retrained when the path or row count changes.

### Feature Engineering

| Feature | Description |
|---|---|
| `balance_error_orig` | Difference between amount and origin balance change — flags inconsistencies |
| `balance_error_dest` | Same for destination — detects money that "appears from nowhere" |
| `dest_is_merchant` | 1 if destination starts with `M` (merchant) |
| `dest_is_customer` | 1 if destination starts with `C` (customer account) |
| `high_value` | 1 if amount > 95th percentile of training data |
| `empties_account` | 1 if `newbalanceOrig == 0` after transaction |
| `type_encoded` | Label-encoded transaction type |

### Scoring (Tab 2)

Upload any CSV with the required columns and the model scores every row in real time:

| Output column | Description |
|---|---|
| `fraud_probability` | Model confidence score [0–1] |
| `fraud_pred` | Binary prediction using the active decision threshold |
| `fraud_pred_model` | Raw model prediction at default threshold |
| `risk_level` | `Bajo` / `Medio` / `Alto` / `Crítico` |

**Risk level thresholds:**

| Level | Probability range |
|---|---|
| Bajo | 0.00 – 0.05 |
| Medio | 0.05 – 0.20 |
| Alto | 0.20 – 0.50 |
| Crítico | 0.50 – 1.00 |

If the uploaded CSV includes the `isFraud` column, the app automatically computes Precision, Recall and F1 against the ground truth.

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run fraud_realtime_dashboard_v4_demo_scoring.py
```

The sidebar lets you configure:
- **Training CSV path** — defaults to `PS_20174392719_1491204439457_log.csv` in the project root
- **Training rows** — how many rows to load (50k–1M, default 250k)
- **Visualization rows** — subsample for interactive charts (default 25k)
- **Decision threshold** — slider 0.01–0.90 (default 0.20)

For a quick local test without the full dataset, point the training path to `paysim_sample_500.csv`.

---

## ☁️ Deployment — Streamlit Community Cloud

1. Push the repo to GitHub (full dataset via Git LFS)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch and set main file:
   ```
   fraud_realtime_dashboard_v4_demo_scoring.py
   ```
4. Deploy — no secrets required (no external API calls)

### Git LFS setup for the full dataset

```bash
git lfs install
git lfs track "PS_20174392719_1491204439457_log.csv"
git add .gitattributes
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## 📊 Demo CSVs

Two lightweight files are included for testing the scoring tab without the full dataset:

| File | Rows | Fraud cases | Notes |
|---|---|---|---|
| `paysim_sample_500.csv` | 500 | 100 (20%) | Real rows sampled from PaySim |
| `paysim_synthetic_1000.csv` | 1000 | 198 (19.8%) | Synthetically generated — respects all original distributions and fraud patterns |

The synthetic dataset was generated by fitting log-normal distributions per transaction type and replicating the key fraud signals: `oldbalanceOrg == amount`, `newbalanceOrig == 0`, `oldbalanceDest == 0` for TRANSFER frauds, etc.

---

## 📋 Required CSV Columns

```
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
nameDest, oldbalanceDest, newbalanceDest, isFraud (optional for scoring)
```

---

## 🛠️ Tech Stack

![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue)
![Git LFS](https://img.shields.io/badge/Git-LFS-green)

---

## 📄 License

This project is for educational and portfolio purposes.

---

*Built as part of a hands-on machine learning and data visualization practice — March 2026*
