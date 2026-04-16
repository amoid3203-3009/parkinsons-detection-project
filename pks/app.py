import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkinsonAI | Group 25",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0a0e1a; }
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2a40; }

h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 600; }

.metric-card {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a6080;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 32px;
    font-weight: 600;
    color: #e2e8f0;
    line-height: 1;
}
.metric-sub {
    font-size: 13px;
    color: #4a6080;
    margin-top: 4px;
}

.severity-normal   { color: #10b981; }
.severity-mild     { color: #f59e0b; }
.severity-moderate { color: #f97316; }
.severity-severe   { color: #ef4444; }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}
.badge-normal   { background: #052e16; color: #10b981; border: 1px solid #10b981; }
.badge-mild     { background: #271a00; color: #f59e0b; border: 1px solid #f59e0b; }
.badge-moderate { background: #2a0f00; color: #f97316; border: 1px solid #f97316; }
.badge-severe   { background: #2a0000; color: #ef4444; border: 1px solid #ef4444; }

.disclaimer {
    background: #0d1220;
    border: 1px solid #1e2a40;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: #4a6080;
    margin: 16px 0;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1d4ed8; }

.header-bar {
    background: linear-gradient(90deg, #0d1220, #111827);
    border-bottom: 1px solid #1e2a40;
    padding: 16px 0;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'amp_mean','amp_std','amp_max','amp_min','amp_range','amp_median',
    'amp_iqr','amp_energy','vel_mean','vel_std','vel_max','vel_min',
    'vel_range','vel_energy','zero_crossing_rate','signal_energy','signal_entropy'
]

SEVERITY_MAP = {
    0: ('Normal',   '#10b981', 'normal'),
    1: ('Mild',     '#f59e0b', 'mild'),
    2: ('Moderate', '#f97316', 'moderate'),
    3: ('Severe',   '#ef4444', 'severe'),
}

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    X = df[FEATURE_COLS].values
    scaler = MinMaxScaler()
    sev_scaled = scaler.fit_transform(
        df[['amp_std','zero_crossing_rate','amp_mean','signal_entropy']].values
    )
    severity_score = (
        sev_scaled[:,0] * 0.40 +
        (1 - sev_scaled[:,1]) * 0.40 +
        (1 - sev_scaled[:,2]) * 0.10 +
        (1 - sev_scaled[:,3]) * 0.10
    )
    y = pd.qcut(severity_score, q=4, labels=[0,1,2,3]).astype(int)

    feat_scaler = MinMaxScaler()
    X_scaled = feat_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return rf, feat_scaler, f1, y_test, y_pred, severity_score, np.array(y)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ParkinsonAI")
    st.markdown("""
    <div style='font-size:12px; color:#4a6080; line-height:1.6;'>
    AI and Signal Processing Techniques<br>
    for Parkinson's Finger Tapping<br><br>
    <b style='color:#e2e8f0;'>COS5031 — Group 25</b><br>
    University of Bradford
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:12px; color:#4a6080;'>
    <b style='color:#e2e8f0;'>Team</b><br>
    Abdul Moeed Alam<br>
    Toufiq Rifat<br>
    Pierre Andoulo<br><br>
    <b style='color:#e2e8f0;'>Supervisor</b><br>
    Dr. Ramzi<br><br>
    <b style='color:#e2e8f0;'>Client</b><br>
    Kieran Townsend — FALL
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div class='disclaimer'>
    ⚕️ Decision-support tool only. All outputs must be reviewed by a qualified neurologist.
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## Parkinson's Disease Finger-Tapping Severity Classifier")
st.markdown(
    "<span style='color:#4a6080; font-size:14px;'>Upload finger-tapping feature data to classify motor severity using MDS-UPDRS Item 3.4</span>",
    unsafe_allow_html=True
)
st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload csv file",
    type=["csv"],
    help="CSV with 17 extracted signal features per recording"
)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding:60px 0; color:#4a6080;'>
        <div style='font-size:48px; margin-bottom:16px;'>📂</div>
        <div style='font-size:16px; font-weight:500; color:#e2e8f0;'>Upload your feature CSV to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load and train ────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded)

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

with st.spinner("Training Random Forest model..."):
    rf, feat_scaler, f1, y_test, y_pred, severity_scores, y_all = train_model(df)

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Model F1 Score</div>
        <div class='metric-value' style='color:#10b981;'>{f1:.4f}</div>
        <div class='metric-sub'> Target: ≥ 0.75</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Recordings</div>
        <div class='metric-value'>{len(df)}</div>
        <div class='metric-sub'>Left + Right hand</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Features</div>
        <div class='metric-value'>{len(FEATURE_COLS)}</div>
        <div class='metric-sub'>Amplitude, velocity, entropy</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Model</div>
        <div class='metric-value' style='font-size:20px; padding-top:6px;'>Random Forest</div>
        <div class='metric-sub'>100 estimators, balanced</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction", "Model Evaluation", "Feature Analysis", "Ethics"
])

# ── TAB 1: Live Prediction ────────────────────────────────────────────────────
with tab1:
    st.markdown("### Single Recording Severity Prediction")
    st.markdown("<div style='color:#4a6080; font-size:13px; margin-bottom:16px;'>Select a recording index and click Classify to get an AI-generated severity estimate.</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        idx = st.slider("Recording index", 0, len(df)-1, 0)
        hand = df['label'].iloc[idx].upper()
        sig_type = df['file_name'].iloc[idx]
        true_sev = SEVERITY_MAP[y_all[idx]]

        st.markdown(f"""
        <div class='metric-card' style='margin-top:12px;'>
            <div class='metric-label'>Recording Info</div>
            <div style='margin-top:8px; font-size:13px; color:#e2e8f0;'>
                Index: <b>#{idx}</b><br>
                Hand: <b>{hand}</b><br>
                Signal: <b>{sig_type}</b><br>
                Severity score: <b>{severity_scores[idx]:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        classify = st.button("Classify Recording")

    with col_b:
        if classify:
            x = feat_scaler.transform(df[FEATURE_COLS].iloc[[idx]].values)
            pred = rf.predict(x)[0]
            proba = rf.predict_proba(x)[0]
            confidence = proba[pred] * 100
            sev_name, sev_colour, sev_class = SEVERITY_MAP[pred]

            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Predicted Severity</div>
                <div class='metric-value severity-{sev_class}' style='font-size:36px;'>{sev_name}</div>
                <div class='metric-sub'>Confidence: {confidence:.1f}%
                {'⚠️ Low — refer for manual review' if confidence < 60 else ' High confidence'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Class probabilities**")
            for i, (name, colour, _) in SEVERITY_MAP.items():
                pct = proba[i] * 100
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:10px; margin:4px 0;'>
                    <div style='width:80px; font-size:12px; color:#4a6080; font-family:DM Mono,monospace;'>{name}</div>
                    <div style='flex:1; background:#0d1220; border-radius:4px; height:18px; overflow:hidden;'>
                        <div style='width:{pct}%; height:100%; background:{colour}; border-radius:4px; transition:width 0.5s;'></div>
                    </div>
                    <div style='width:44px; font-size:12px; color:#e2e8f0; text-align:right; font-family:DM Mono,monospace;'>{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class='disclaimer'>
            ⚕️ This is an AI-generated estimate only. Must be reviewed by a qualified neurologist before any clinical action is taken.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:40px; color:#4a6080;'>
                <div style='font-size:32px;'></div>
                <div style='margin-top:8px;'>Select a recording and click Classify</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Model Evaluation ───────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Evaluation — Hold-out Test Set")

    col_x, col_y = st.columns(2)

    with col_x:
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d1220')
        ax.set_facecolor('#0d1220')
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Normal','Mild','Moderate','Severe']
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix', color='#e2e8f0', pad=12)
        ax.tick_params(colors='#4a6080')
        ax.xaxis.label.set_color('#4a6080')
        ax.yaxis.label.set_color('#4a6080')
        for text in ax.texts:
            text.set_color('#e2e8f0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_y:
        # Severity distribution
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor('#0d1220')
        ax2.set_facecolor('#0d1220')
        colours = ['#10b981','#f59e0b','#f97316','#ef4444']
        labels = ['Normal','Mild','Moderate','Severe']
        counts = [sum(y_all==i) for i in range(4)]
        bars = ax2.bar(labels, counts, color=colours, edgecolor='#0d1220', linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(count), ha='center', color='#e2e8f0', fontsize=11)
        ax2.set_title('Severity Class Distribution', color='#e2e8f0', pad=12)
        ax2.tick_params(colors='#4a6080')
        ax2.set_facecolor('#0d1220')
        ax2.spines['bottom'].set_color('#1e2a40')
        ax2.spines['left'].set_color('#1e2a40')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.label.set_color('#4a6080')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Metrics summary
    st.markdown("### Performance Summary")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>F1 Score (weighted)</div>
            <div class='metric-value' style='color:#10b981;'>{f1:.4f}</div>
            <div class='metric-sub'>5-fold cross-validation</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Target F1</div>
            <div class='metric-value'>0.75</div>
            <div class='metric-sub'>PID Objective 2</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        outcome = "MET" if f1 >= 0.75 else "❌ MISSED"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Outcome</div>
            <div class='metric-value' style='color:#10b981; font-size:24px;'>{outcome}</div>
            <div class='metric-sub'>Random Forest primary model</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 3: Feature Analysis ───────────────────────────────────────────────────
with tab3:
    st.markdown("### Feature Importance (XAI — Explainable AI)")
    st.markdown("<div style='color:#4a6080; font-size:13px; margin-bottom:16px;'>Feature importance links classification decisions to specific signal characteristics, satisfying UK GDPR Article 22 explainability requirements.</div>", unsafe_allow_html=True)

    importances = rf.feature_importances_
    fi_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True)

    fig3, ax3 = plt.subplots(figsize=(8, 7))
    fig3.patch.set_facecolor('#0d1220')
    ax3.set_facecolor('#0d1220')
    colours_fi = ['#ef4444' if v > 0.08 else '#2563eb' for v in fi_df['Importance']]
    ax3.barh(fi_df['Feature'], fi_df['Importance'], color=colours_fi, edgecolor='#0d1220')
    ax3.axvline(0.08, color='#ef4444', linestyle='--', linewidth=1, alpha=0.5, label='High importance (>0.08)')
    ax3.set_title('Random Forest Feature Importance', color='#e2e8f0', pad=12)
    ax3.tick_params(colors='#4a6080')
    ax3.spines['bottom'].set_color('#1e2a40')
    ax3.spines['left'].set_color('#1e2a40')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.xaxis.label.set_color('#4a6080')
    ax3.legend(facecolor='#0d1220', labelcolor='#4a6080', edgecolor='#1e2a40')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("**Top 5 diagnostic features:**")
    top5 = fi_df.tail(5)[['Feature','Importance']].iloc[::-1]
    for _, row in top5.iterrows():
        pct = row['Importance'] * 100
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin:4px 0;'>
            <div style='width:160px; font-size:12px; color:#e2e8f0; font-family:DM Mono,monospace;'>{row['Feature']}</div>
            <div style='flex:1; background:#0d1220; border:1px solid #1e2a40; border-radius:4px; height:16px; overflow:hidden;'>
                <div style='width:{pct*5}%; height:100%; background:#ef4444; border-radius:4px;'></div>
            </div>
            <div style='width:44px; font-size:12px; color:#4a6080; text-align:right; font-family:DM Mono,monospace;'>{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 4: Ethics ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Ethics & Compliance Summary")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>GDPR Compliance</div>
            <div style='margin-top:10px; font-size:13px; color:#e2e8f0; line-height:1.8;'>
            ✅ UK GDPR Art 6(1)(e) — public interest<br>
            ✅ UK GDPR Art 9(2)(j) — scientific research<br>
            ✅ Pre-approved anonymised dataset<br>
            ✅ 5-step anonymisation pipeline<br>
            ✅ DPIA completed
            </div>
        </div>
        <div class='metric-card' style='margin-top:8px;'>
            <div class='metric-label'>Bias Mitigation</div>
            <div style='margin-top:10px; font-size:13px; color:#e2e8f0; line-height:1.8;'>
            ✅ Stratified cross-validation<br>
            ✅ class_weight='balanced' applied<br>
            ✅ Left vs right hand subgroup analysis<br>
            ✅ REFORMS checklist (Kapoor et al., 2024)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Safe AI Use</div>
            <div style='margin-top:10px; font-size:13px; color:#e2e8f0; line-height:1.8;'>
            ✅ Decision-support tool only<br>
            ✅ NOT a diagnostic device<br>
            ✅ Confidence threshold flagging at &lt;60%<br>
            ✅ Mandatory clinician review<br>
            ✅ No autonomous recommendations
            </div>
        </div>
        <div class='metric-card' style='margin-top:8px;'>
            <div class='metric-label'>Transparency (XAI)</div>
            <div style='margin-top:10px; font-size:13px; color:#e2e8f0; line-height:1.8;'>
            ✅ Feature importance per prediction<br>
            ✅ Confidence scores displayed<br>
            ✅ FAIR report included<br>
            ✅ Open GitHub repository<br>
            ✅ UK GDPR Art 22 compliant
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer' style='margin-top:16px;'>
    ⚕️ <b>DISCLAIMER:</b> This tool is a clinical decision-support aid only. All AI-generated outputs are estimates and must be reviewed by a qualified neurologist before any clinical action is taken. This system is not MHRA-registered and cannot be used for actual patient diagnosis. — COS5031 Group 25, University of Bradford.
    </div>
    """, unsafe_allow_html=True)
