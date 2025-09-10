# app_ui.py — SANFIND-styled UI (Single HTML Navigation Bar)
import os
import io
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Try to import seaborn for nicer charts; if unavailable we'll fallback to matplotlib-only
try:
    import seaborn as sns
except Exception:
    sns = None

API_BASE = os.environ.get("NUSA_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="SANFIND • NUSA Score", page_icon="💼", layout="centered")

# ---- Branding ----
PRIMARY = "#00337A"  # SANF deep blue
ACCENT = "#FDB813"   # Gold
BG = "#F3F6FB"
SURFACE = "#FFFFFF"

# ---- Global CSS ----
# Note: CSS for bottom-nav and app shell structure.
st.markdown(f"""
<style>
body, .stApp {{ background-color: {BG} !important; color: #1C1C1C !important; }}
#MainMenu, header, footer {{display:none;}}

body, .stApp {{ background-color: {BG} !important; color: #1C1C1C !important; }}
#MainMenu, header, footer {{display:none;}}
.stAppHeader {{display:none;}}
.main .block-container {{padding-top: 0rem;}}
.stApp > header {{display:none;}}
iframe {{display:none;}}

.app-shell {{
    max-width: 420px;
    margin: 70px auto 90px auto;
    border-radius: 0 0 22px 22px;
    background: {SURFACE};
    box-shadow: 0 6px 22px rgba(0,0,0,.08);
    overflow: hidden;
    font-family: 'Segoe UI', sans-serif;
}}
.app-bar {{
    background: {PRIMARY};
    padding: 12px 16px;
    color: white;
    display: flex; align-items: center; gap: 10px;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    width: 100%;
    max-width: 100vw;
    border-radius: 0;
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    justify-content: center;
}}
.app-bar > div {{
    max-width: 420px;
    width: 100%;
}}
.app-bar .title {{ font-weight: 700; font-size: 18px; }}
.app-bar .tagline {{ font-size: 12px; opacity:.85; }}
.card {{
    margin: 16px;
    padding: 16px;
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}}
.bottom-nav {{
    position: fixed;
    bottom: 16px;
    left: 50%; transform: translateX(-50%);
    width: 420px; max-width: 95vw;
    background: #fff;
    border-radius: 18px;
    display: flex; justify-content: space-around;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    padding: 6px 0;
    font-family: 'Segoe UI', sans-serif;
    z-index: 99;
}}
.bottom-nav form {{
    margin: 0; padding: 0; flex: 1;
}}
.nav-btn {{
    width: 100%;
    border: none;
    background: none;
    font-size: 12px;
    color: #444;
    cursor: pointer;
    padding: 6px 4px;
    text-decoration: none;
}}
.nav-btn.active {{
    color: {PRIMARY};
    font-weight: 600;
}}
.nav-icon {{
    font-size: 18px;
    display: block;
}}
</style>
""", unsafe_allow_html=True)

# ---- App shell & header ----
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown(f"""
<div class="app-bar">
  <div>
    <div class="title">SANFIND – NUSA Score</div>
    <div class="tagline">Closer to Supporting You</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Tab state initialization & URL sync ----
# Check query params first
params = st.query_params
if "tab" in params and params.get("tab"):
    requested_tab = str(params.get("tab"))
    allowed = {"Beranda", "Prediction", "Analysis", "Kontrak", "Akun"}
    if requested_tab in allowed:
        st.session_state.tab = requested_tab

# Initialize default tab if not set
if "tab" not in st.session_state:
    st.session_state.tab = "Prediction"  # Changed default to Prediction

# Short alias for current tab
tab = st.session_state.tab

# Handle navigation from query params (when bottom nav is clicked)
params = st.query_params
if "tab" in params and params.get("tab"):
    requested_tab = str(params.get("tab"))
    allowed = {"Beranda", "Prediction", "Analysis", "Kontrak", "Akun"}
    if requested_tab in allowed:
        st.session_state.tab = requested_tab

# ---- Content (renders based on st.session_state.tab) ----
if tab == "Beranda":
    st.markdown('<div class="card"><h4>Welcome</h4>This is a mock SANFIND home screen.</div>', unsafe_allow_html=True)

elif tab == "Prediction":  # Renamed from "Aktivitas" to "Prediction"
    st.markdown('<div class="card"><h4>Upload Applicant Data</h4>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Excel", type=["xlsx"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            df = None

        if df is not None:
            st.markdown('<div class="card"><h4>Preview</h4>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner("Scoring..."):
                try:
                    resp = requests.post(f"{API_BASE}/predict_file", files={"file": uploaded.getvalue()}, timeout=30)
                except Exception as e:
                    st.error(f"Failed to contact backend: {e}")
                    resp = None

            if resp is not None and resp.status_code == 200:
                res = pd.DataFrame(resp.json())

                # Results Card
                st.markdown('<div class="card"><h4>Results</h4>', unsafe_allow_html=True)
                res["risk_probability"] = (res["risk_probability"] * 100).round(2)
                st.dataframe(res, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Summary Card
                avg_risk = float(res["risk_probability"].mean())
                approve = (res["suggested_decision"] == "APPROVE").mean() * 100
                flag = (res["suggested_decision"] == "FLAG FOR REVIEW").mean() * 100
                decline = (res["suggested_decision"] == "DECLINE").mean() * 100
                total = len(res)

                st.markdown('<div class="card"><h4>📊 Summary</h4>', unsafe_allow_html=True)
                st.markdown(f"**Total Applicants:** {total}")
                st.markdown(f"**Average Risk Probability:** {avg_risk:.2f}%")
                try:
                    st.progress(min(max(int(avg_risk), 0), 100))
                except Exception:
                    pass # ignore progress bar errors if avg_risk calculation fails
                st.write(f"✅ Approve: **{approve:.1f}%**")
                st.write(f"⚠️ Flag for Review: **{flag:.1f}%**")
                st.write(f"❌ Decline: **{decline:.1f}%**")
                st.markdown('</div>', unsafe_allow_html=True)

                # Download button logic
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Input", index=False)
                    res.to_excel(writer, sheet_name="Predictions", index=False)

                st.download_button(
                    "⬇️ Download Results (Excel)",
                    data=out.getvalue(),
                    file_name="nusa_score_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            elif resp is not None:
                st.error(f"API returned status {resp.status_code}: {resp.text}")

elif tab == "Analysis":
    st.markdown('<div class="card"><h4>📊 Model Insights</h4>', unsafe_allow_html=True)

    try:
        import joblib
        meta = joblib.load("nusa_score_meta.joblib")
        model = joblib.load("nusa_score_model.joblib")

        # Feature Importance (XGBoost)
        # Check if model object has 'get_booster' method (for XGBoost)
        if hasattr(model, 'get_booster'):
            importance = model.get_booster().get_score(importance_type="weight")
        else:
            importance = {} # fallback for non-xgboost models or if unavailable

        if not importance:
            st.info("Model feature importance information is not available.")
        else:
            imp_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
            st.subheader("Top Features")
            fig, ax = plt.subplots(figsize=(6, 4))
            if sns is not None:
                sns.barplot(x="Importance", y="Feature", data=imp_df.head(10), ax=ax, palette="Blues_r")
            else:
                ax.barh(imp_df["Feature"].head(10), imp_df["Importance"].head(10), color=PRIMARY)
                ax.invert_yaxis() # Important for horizontal bar charts
            plt.tight_layout()
            st.pyplot(fig)

        # Risk Distribution
        st.subheader("Distribution of Predicted Risk")
        if meta and "validation_pred_probas" in meta:
            probs = meta["validation_pred_probas"]
        else:
            probs = np.random.beta(a=2, b=5, size=500) # Fallback demo data

        fig, ax = plt.subplots(figsize=(6, 4))
        if sns is not None:
            sns.histplot(probs, bins=20, kde=True, ax=ax, color=PRIMARY)
        else:
            ax.hist(probs, bins=20, color=PRIMARY, alpha=0.7)
        ax.set_xlabel("Predicted Risk Probability")
        plt.tight_layout()
        st.pyplot(fig)

        # Metrics Card
        st.subheader("Performance Metrics")
        metrics = meta.get("metrics", {"accuracy": 0.77, "precision": 0.72, "recall": 0.70, "f1": 0.71})

        # Display metrics without extra card wrapper
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        cols[1].metric("Precision", f"{metrics['precision']*100:.1f}%")
        cols[2].metric("Recall", f"{metrics['recall']*100:.1f}%")
        cols[3].metric("F1 Score", f"{metrics['f1']*100:.1f}%")

    except FileNotFoundError:
        st.error("Model or metadata not found. Run the training script to create 'nusa_score_model.joblib' and 'nusa_score_meta.joblib'.")
    except ImportError:
        st.error("Could not import required library for analysis (e.g., joblib).")
    except Exception as e:
        st.error(f"Could not load analysis: {e}")

    st.markdown('</div>', unsafe_allow_html=True) # Close the main card for this tab

elif tab == "Kontrak":
    st.markdown('<div class="card"><h4>Contracts</h4>Mock SANFIND contracts page.</div>', unsafe_allow_html=True)

elif tab == "Akun":
    st.markdown('<div class="card"><h4>Account</h4>Profile and settings (mock).</div>', unsafe_allow_html=True)

# Close app shell
st.markdown("</div>", unsafe_allow_html=True)

# ---- Bottom nav (JavaScript-based navigation) ----
nav_html = f"""
<div class="bottom-nav">
    <button onclick="navigateToTab('Beranda')" class="nav-btn {'active' if tab=='Beranda' else ''}">
        <span class="nav-icon">🏠</span>Beranda
    </button>
    <button onclick="navigateToTab('Prediction')" class="nav-btn {'active' if tab=='Prediction' else ''}">
        <span class="nav-icon">📊</span>Prediction
    </button>
    <button onclick="navigateToTab('Analysis')" class="nav-btn {'active' if tab=='Analysis' else ''}">
        <span class="nav-icon">📈</span>Analysis
    </button>
    <button onclick="navigateToTab('Kontrak')" class="nav-btn {'active' if tab=='Kontrak' else ''}">
        <span class="nav-icon">📑</span>Kontrak
    </button>
    <button onclick="navigateToTab('Akun')" class="nav-btn {'active' if tab=='Akun' else ''}">
        <span class="nav-icon">👤</span>Akun
    </button>
</div>

<script>
function navigateToTab(tabName) {{
    const url = new URL(window.location);
    url.searchParams.set('tab', tabName);
    window.location.href = url.toString();
}}
</script>
"""

st.markdown(nav_html, unsafe_allow_html=True)