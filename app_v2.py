import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import savgol_filter
from collections import deque

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkinsonAI v2 | Group 25",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2a40; }
.metric-card {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid #1e2a40; border-radius: 12px;
    padding: 20px 24px; margin: 8px 0;
}
.metric-label {
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: #4a6080; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px;
}
.metric-value { font-size: 32px; font-weight: 600; color: #e2e8f0; line-height: 1; }
.metric-sub { font-size: 13px; color: #4a6080; margin-top: 4px; }
.disclaimer {
    background: #0d1220; border: 1px solid #1e2a40;
    border-left: 3px solid #3b82f6; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: #4a6080; margin: 16px 0;
}
.live-badge {
    background: #052e16; color: #10b981; border: 1px solid #10b981;
    border-radius: 20px; padding: 4px 12px; font-size: 12px;
    font-family: 'DM Mono', monospace; display: inline-block;
}
.stButton > button {
    background: #2563eb; color: white; border: none;
    border-radius: 8px; padding: 10px 24px;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 14px; width: 100%;
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
    0: ('Normal',   '#10b981'),
    1: ('Mild',     '#f59e0b'),
    2: ('Moderate', '#f97316'),
    3: ('Severe',   '#ef4444'),
}

CSV_PATH = 'data/raw/finger_tapping_features.csv'

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return None, None, None, None

    X = df[FEATURE_COLS].values
    sev_scaler = MinMaxScaler()
    sev_scaled = sev_scaler.fit_transform(
        df[['amp_std','zero_crossing_rate','amp_mean','signal_entropy']].values
    )
    severity_score = (
        sev_scaled[:,0] * 0.40 + (1 - sev_scaled[:,1]) * 0.40 +
        (1 - sev_scaled[:,2]) * 0.10 + (1 - sev_scaled[:,3]) * 0.10
    )
    y = pd.qcut(severity_score, q=4, labels=[0,1,2,3]).astype(int)

    feat_scaler = MinMaxScaler()
    X_scaled = feat_scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf, feat_scaler, df, np.array(y)

# ── Feature extraction from raw signal ───────────────────────────────────────
def extract_features_from_signal(signal: np.ndarray) -> np.ndarray:
    """Extract the same 17 features Toufiq's pipeline produces."""
    if len(signal) < 5:
        return None

    # Clean signal
    q1, q3 = np.percentile(signal, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        signal = np.clip(signal, q1 - 1.5*iqr, q3 + 1.5*iqr)
    win = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win >= 3:
        signal = savgol_filter(signal, window_length=win, polyorder=2)

    # Velocity
    velocity = np.diff(signal, prepend=signal[0])

    # Features
    amp_mean   = np.mean(signal)
    amp_std    = np.std(signal)
    amp_max    = np.max(signal)
    amp_min    = np.min(signal)
    amp_range  = amp_max - amp_min
    amp_median = np.median(signal)
    amp_iqr    = float(np.percentile(signal, 75) - np.percentile(signal, 25))
    amp_energy = float(np.sum(signal ** 2))

    vel_mean   = np.mean(velocity)
    vel_std    = np.std(velocity)
    vel_max    = np.max(velocity)
    vel_min    = np.min(velocity)
    vel_range  = vel_max - vel_min
    vel_energy = float(np.sum(velocity ** 2))

    zero_crossings = np.where(np.diff(np.sign(signal - np.mean(signal))))[0]
    zcr = len(zero_crossings) / len(signal)

    sig_energy  = amp_energy
    prob = signal / (signal.sum() + 1e-10)
    entropy = -np.sum(prob * np.log(prob + 1e-10))

    return np.array([[
        amp_mean, amp_std, amp_max, amp_min, amp_range,
        amp_median, amp_iqr, amp_energy,
        vel_mean, vel_std, vel_max, vel_min, vel_range, vel_energy,
        zcr, sig_energy, entropy
    ]])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ParkinsonAI **v2**")
    st.markdown("""
    <div class='live-badge'>● LIVE DETECTION</div>
    <div style='font-size:12px; color:#4a6080; line-height:1.6; margin-top:12px;'>
    COS5031 — Group 25<br>
    University of Bradford<br><br>
    <b style='color:#e2e8f0;'>New in v2:</b><br>
    Real-time camera detection<br>
    using finger-tap tracking
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:12px; color:#4a6080;'>
    <b style='color:#e2e8f0;'>Team</b><br>
    Abdul Moeed Alam<br>Toufiq Rifat<br>Pierre Andoulo
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div class='disclaimer'>
    ⚕️ Decision-support only. All outputs must be reviewed by a qualified neurologist.
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## ParkinsonAI v2 — Live Camera Detection")
st.markdown(
    "<span style='color:#4a6080; font-size:14px;'>Point your camera at a finger-tapping movement — the system extracts features in real time and classifies severity.</span>",
    unsafe_allow_html=True
)
st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
rf, feat_scaler, df_ref, y_ref = train_model()

if rf is None:
    st.error("Could not load training data. Make sure `data/raw/finger_tapping_features.csv` exists.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Live Camera Detection", "CSV Analysis (v1)", "⚕️ Ethics"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE CAMERA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Real-Time Finger-Tapping Severity Detection")
    st.markdown("""
    <div style='color:#4a6080; font-size:13px; margin-bottom:16px;'>
    How to use: Allow camera access → tap your index finger against your thumb repeatedly for 5 seconds → click Analyse.
    </div>
    """, unsafe_allow_html=True)

    col_inst, col_result = st.columns([1, 1])

    with col_inst:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Instructions</div>
            <div style='font-size:13px; color:#e2e8f0; line-height:2; margin-top:8px;'>
            1. Click <b>Start Recording</b><br>
            2. Tap index finger to thumb rapidly<br>
            3. Keep hand visible in the frame<br>
            4. After 5 seconds click <b>Analyse</b><br>
            5. Results appear on the right
            </div>
        </div>
        """, unsafe_allow_html=True)

        duration = st.slider("Recording duration (seconds)", 3, 10, 5)
        fps_target = 15

    with col_result:
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div style='text-align:center; padding:40px; color:#4a6080;'>
            <div style='font-size:32px;'>📷</div>
            <div style='margin-top:8px;'>Results will appear here after analysis</div>
        </div>
        """, unsafe_allow_html=True)

    # Camera capture
    st.markdown("---")
    img_placeholder   = st.empty()
    status_placeholder = st.empty()
    progress_bar      = st.empty()

    col_start, col_analyse = st.columns(2)

    with col_start:
        start = st.button("▶ Start Recording")
    with col_analyse:
        analyse = st.button("Analyse Recording")

    # Session state
    if 'signal_buffer' not in st.session_state:
        st.session_state.signal_buffer = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'frames_captured' not in st.session_state:
        st.session_state.frames_captured = []

    if start:
        st.session_state.signal_buffer = []
        st.session_state.frames_captured = []
        st.session_state.recording = True

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access camera. Please allow camera permission and try again.")
            st.session_state.recording = False
        else:
            total_frames = duration * fps_target
            frame_count = 0
            signal_values = []

            status_placeholder.markdown(
                "<div style='color:#10b981; font-size:13px;'>● Recording... tap your finger now</div>",
                unsafe_allow_html=True
            )

            start_time = time.time()
            prev_y = None

            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]

                # Draw recording indicator
                cv2.circle(frame_rgb, (30, 30), 10, (16, 185, 129), -1)
                cv2.putText(frame_rgb, "RECORDING", (50, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2)

                # Simple brightness-based motion proxy in hand region
                # (centre-bottom quarter of frame where hand typically is)
                roi = frame_rgb[h//2:, w//4:3*w//4]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                mean_brightness = float(np.mean(gray_roi)) / 255.0

                if prev_y is not None:
                    signal_values.append(abs(mean_brightness - prev_y))
                prev_y = mean_brightness

                frame_count += 1
                elapsed = time.time() - start_time
                pct = min(elapsed / duration, 1.0)

                img_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                progress_bar.progress(pct, text=f"Recording: {elapsed:.1f}s / {duration}s")

                time.sleep(1.0 / fps_target)

            cap.release()
            st.session_state.signal_buffer = signal_values
            st.session_state.recording = False
            status_placeholder.markdown(
                "<div style='color:#f59e0b; font-size:13px;'>● Recording complete — click Analyse</div>",
                unsafe_allow_html=True
            )
            progress_bar.progress(1.0, text="Done")

    if analyse and len(st.session_state.signal_buffer) > 10:
        signal = np.array(st.session_state.signal_buffer)

        features = extract_features_from_signal(signal)

        if features is not None:
            # Scale and predict
            feat_scaled = feat_scaler.transform(features)
            pred = rf.predict(feat_scaled)[0]
            proba = rf.predict_proba(feat_scaled)[0]
            confidence = proba[pred] * 100
            sev_name, sev_colour = SEVERITY_MAP[pred]

            with col_result:
                result_placeholder.empty()
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Live Detection Result</div>
                    <div style='font-size:36px; font-weight:600; color:{sev_colour}; margin:8px 0;'>{sev_name}</div>
                    <div style='font-size:13px; color:#4a6080;'>Confidence: {confidence:.1f}%
                    {"Low ⚠️ — refer for review" if confidence < 60 else "High confidence"}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Class probabilities**")
                for i, (name, colour) in SEVERITY_MAP.items():
                    pct = proba[i] * 100
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; gap:10px; margin:4px 0;'>
                        <div style='width:80px; font-size:12px; color:#4a6080; font-family:DM Mono,monospace;'>{name}</div>
                        <div style='flex:1; background:#0d1220; border-radius:4px; height:18px; overflow:hidden;'>
                            <div style='width:{pct}%; height:100%; background:{colour}; border-radius:4px;'></div>
                        </div>
                        <div style='width:44px; font-size:12px; color:#e2e8f0; text-align:right;'>{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Signal plot
            st.markdown("**Captured finger-tapping signal**")
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            fig.patch.set_facecolor('#0d1220')
            for ax in axes:
                ax.set_facecolor('#0d1220')
                ax.spines['bottom'].set_color('#1e2a40')
                ax.spines['left'].set_color('#1e2a40')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors='#4a6080')

            axes[0].plot(signal, color='#d62728', linewidth=0.8, alpha=0.8)
            axes[0].set_title('Raw Captured Signal', color='#e2e8f0')
            axes[0].set_ylabel('Motion value', color='#4a6080')

            # Cleaned
            cleaned = signal.copy()
            q1, q3 = np.percentile(cleaned, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                cleaned = np.clip(cleaned, q1-1.5*iqr, q3+1.5*iqr)
            win = min(11, len(cleaned) if len(cleaned) % 2 == 1 else len(cleaned)-1)
            if win >= 3:
                cleaned = savgol_filter(cleaned, window_length=win, polyorder=2)

            axes[1].plot(cleaned, color='#1f77b4', linewidth=0.8)
            axes[1].set_title('Cleaned + Smoothed Signal', color='#e2e8f0')
            axes[1].set_ylabel('Motion value', color='#4a6080')
            axes[1].set_xlabel('Frame', color='#4a6080')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            <div class='disclaimer'>
            ⚕️ This is an AI-generated estimate from live camera input. Signal quality depends on lighting
            and camera stability. Must be reviewed by a qualified neurologist before any clinical action.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Not enough signal captured. Try recording for longer or improve lighting.")

    elif analyse and len(st.session_state.signal_buffer) == 0:
        st.warning("No recording found. Press Start Recording first.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — CSV ANALYSIS (v1 feature)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### CSV-Based Analysis (Version 1 Feature)")
    st.markdown("<div style='color:#4a6080; font-size:13px; margin-bottom:16px;'>Upload a pre-processed feature CSV to classify severity across all recordings.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload finger_tapping_features.csv", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = feat_scaler.transform(df[FEATURE_COLS].values)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                idx = st.slider("Recording index", 0, len(df)-1, 0)
                if st.button("Classify"):
                    x_demo = X[[idx]]
                    pred = rf.predict(x_demo)[0]
                    proba = rf.predict_proba(x_demo)[0]
                    confidence = proba[pred] * 100
                    sev_name, sev_colour = SEVERITY_MAP[pred]

                    with col_b:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Predicted Severity</div>
                            <div style='font-size:36px; font-weight:600; color:{sev_colour};'>{sev_name}</div>
                            <div style='font-size:13px; color:#4a6080;'>Confidence: {confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                        for i, (name, colour) in SEVERITY_MAP.items():
                            pct = proba[i] * 100
                            st.markdown(f"""
                            <div style='display:flex; align-items:center; gap:10px; margin:4px 0;'>
                                <div style='width:80px; font-size:12px; color:#4a6080;'>{name}</div>
                                <div style='flex:1; background:#0d1220; border-radius:4px; height:18px;'>
                                    <div style='width:{pct}%; height:100%; background:{colour}; border-radius:4px;'></div>
                                </div>
                                <div style='width:44px; font-size:12px; color:#e2e8f0; text-align:right;'>{pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — ETHICS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Ethics & Compliance")
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Safe AI Use — Live Camera</div>
        <div style='font-size:13px; color:#e2e8f0; line-height:2; margin-top:8px;'>
        ✅ No video is stored — frames are processed in memory only<br>
        ✅ No biometric data is retained after session ends<br>
        ✅ Camera access requires explicit user permission<br>
        ✅ All predictions are AI-generated estimates only<br>
        ✅ Low-confidence results flagged for manual review<br>
        ✅ Decision-support tool only — not a diagnostic device
        </div>
    </div>
    <div class='disclaimer' style='margin-top:16px;'>
    ⚕️ <b>DISCLAIMER:</b> This tool is a clinical decision-support aid only.
    All AI-generated outputs must be reviewed by a qualified neurologist before any clinical action is taken.
    This system is not MHRA-registered and cannot be used for actual patient diagnosis.
    No video data is stored or transmitted. — COS5031 Group 25, University of Bradford.
    </div>
    """, unsafe_allow_html=True)
