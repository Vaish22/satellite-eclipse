"""
=============================================================
  STREAMLIT DASHBOARD
  AI-Assisted Satellite Eclipse Prediction System
=============================================================
  Run locally:
    streamlit run app.py

  Deploy free on Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → deploy!
=============================================================
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time
from datetime import datetime, timezone

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eclipse·Pred — Satellite Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark space theme */
  .stApp { background-color: #080c12; color: #c9d1d9; }
  section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1c2333; }
  
  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid #1c2333;
    border-radius: 6px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #556 !important; font-family: monospace; font-size: 11px; letter-spacing: 2px; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: monospace; }

  /* Headers */
  h1, h2, h3 { font-family: 'Courier New', monospace !important; color: #00d4ff !important; }
  
  /* Tabs */
  .stTabs [data-baseweb="tab"] { font-family: monospace; font-size: 12px; letter-spacing: 2px; color: #556; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
  
  /* Selectbox */
  .stSelectbox label { color: #556 !important; font-family: monospace; font-size: 11px; }

  /* Info boxes */
  .eclipse-box {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.3);
    border-left: 3px solid #f85149;
    padding: 12px 16px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 13px;
    margin: 8px 0;
  }
  .sunlit-box {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.3);
    border-left: 3px solid #3fb950;
    padding: 12px 16px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 13px;
    margin: 8px 0;
  }
  .info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 3px solid #00d4ff;
    padding: 12px 16px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 13px;
    margin: 8px 0;
  }
  
  /* Hide streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MU     = {"Earth": 3.986004418e14, "Mars": 4.282837e13, "Moon": 4.9048695e12}
R_BODY = {"Earth": 6.371e6,        "Mars": 3.3895e6,    "Moon": 1.7374e6}
R_SUN  = 6.957e8
AU     = 1.495978707e11

MISSIONS = {
    "🌍 ISS — Earth Orbit": {
        "body": "Earth", "color": "#00d4ff",
        "a": 6778e3, "e": 0.001, "inc": np.radians(51.6),
        "raan": 0, "argp": 0, "M0": 0,
        "desc": "407 km · 51.6° inclination · 92.6 min period",
        "battery_start": 85,
    },
    "📡 GPS — MEO Orbit": {
        "body": "Earth", "color": "#ffd700",
        "a": 26560e3, "e": 0.01, "inc": np.radians(55),
        "raan": np.radians(90), "argp": 0, "M0": 0,
        "desc": "20,200 km · 55° inclination · 718 min period",
        "battery_start": 95,
    },
    "🔴 MRO — Mars Orbit": {
        "body": "Mars", "color": "#ff6b35",
        "a": 7189.5e3, "e": 0.005, "inc": np.radians(92.6),
        "raan": 0, "argp": 0, "M0": 0,
        "desc": "3,800 km · 92.6° inclination · near-polar",
        "battery_start": 72,
    },
    "🌕 LRO — Lunar Orbit": {
        "body": "Moon", "color": "#c8b8e8",
        "a": 1787.4e3, "e": 0.001, "inc": np.radians(90),
        "raan": 0, "argp": 0, "M0": 0,
        "desc": "50 km · 90° polar inclination",
        "battery_start": 60,
    },
}

# ─── Physics Functions ────────────────────────────────────────────────────────
def solve_kepler(M, e, tol=1e-10):
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E

def propagate_orbit(m, times):
    a, e, inc = m["a"], m["e"], m["inc"]
    raan, argp, M0 = m["raan"], m["argp"], m["M0"]
    mu = MU[m["body"]]
    n  = np.sqrt(mu / a**3)
    p  = a * (1 - e**2)

    cosO, sinO = np.cos(raan), np.sin(raan)
    cosi, sini = np.cos(inc),  np.sin(inc)
    cosw, sinw = np.cos(argp), np.sin(argp)
    R = np.array([
        [ cosO*cosw - sinO*sinw*cosi, -cosO*sinw - sinO*cosw*cosi,  sinO*sini],
        [ sinO*cosw + cosO*sinw*cosi, -sinO*sinw + cosO*cosw*cosi, -cosO*sini],
        [ sinw*sini,                   cosw*sini,                    cosi     ]
    ])

    positions = np.zeros((len(times), 3))
    for k, t in enumerate(times):
        M  = (M0 + n * t) % (2 * np.pi)
        E  = solve_kepler(M, e)
        nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
        r  = p / (1 + e * np.cos(nu))
        r_pf = r * np.array([np.cos(nu), np.sin(nu), 0])
        positions[k] = R @ r_pf
    return positions

def sun_position(times, body="Earth"):
    d = AU if body != "Mars" else 1.524 * AU
    w = 2 * np.pi / (365.25 * 86400)
    sun = np.zeros((len(times), 3))
    sun[:, 0] = d * np.cos(w * times)
    sun[:, 1] = d * np.sin(w * times)
    return sun

def eclipse_factor(sat_pos, sun_pos, body):
    Rb  = R_BODY[body]
    ef  = np.zeros(len(sat_pos))
    for k in range(len(sat_pos)):
        rs  = np.linalg.norm(sat_pos[k])
        rn  = np.linalg.norm(sun_pos[k])
        dot = np.dot(sat_pos[k], sun_pos[k])
        cos_a = np.clip(dot / (rs * rn), -1, 1)
        angle = np.arccos(cos_a)
        if angle < np.pi / 2:
            continue
        d    = np.linalg.norm(sun_pos[k] - sat_pos[k])
        ab   = np.arcsin(np.clip(Rb / rs, -1, 1))
        as_  = np.arcsin(np.clip(R_SUN / d, -1, 1))
        sep  = np.pi - angle
        if   sep < ab - as_: ef[k] = 1.0
        elif sep < ab + as_: ef[k] = 0.5
    return ef

def run_simulation(m, n_orbits=5, steps=800):
    mu = MU[m["body"]]
    T  = 2 * np.pi * np.sqrt(m["a"]**3 / mu)
    times    = np.linspace(0, n_orbits * T, steps)
    positions = propagate_orbit(m, times)
    sun_pos   = sun_position(times, m["body"])
    ef        = eclipse_factor(positions, sun_pos, m["body"])
    return times, positions, sun_pos, ef, T

def extract_events(times, ef, body):
    events = []
    in_e, start_t, etype = False, 0, "umbra"
    for k in range(len(times)):
        if not in_e and ef[k] > 0:
            in_e = True; start_t = times[k]
            etype = "umbra" if ef[k] >= 1 else "penumbra"
        elif in_e and ef[k] == 0:
            in_e = False
            events.append({"start": start_t/60, "end": times[k]/60,
                           "duration": (times[k]-start_t)/60, "type": etype, "body": body})
    return events

# ─── AI Model ─────────────────────────────────────────────────────────────────
def build_features(m):
    a, e, inc = m["a"], m["e"], m["inc"]
    mu = MU[m["body"]]
    Rb = R_BODY[m["body"]]
    T  = 2 * np.pi * np.sqrt(a**3 / mu)
    return np.array([[
        a/1e6, e, np.degrees(inc), np.degrees(m["raan"]),
        np.degrees(m["argp"]), np.cos(m["M0"]), np.sin(m["M0"]),
        T/3600, (a*(1-e)-Rb)/1e3, (a*(1+e)-Rb)/1e3,
        np.cos(inc), np.sin(inc)
    ]], dtype=np.float32)

@st.cache_resource(show_spinner=False)
def train_model():
    """Train the AI model — cached so it only runs once."""
    np.random.seed(42)
    X, y = [], []
    for _ in range(400):
        body = np.random.choice(["Earth","Mars","Moon"])
        Rb   = R_BODY[body]; mu = MU[body]
        a    = Rb + np.random.uniform(200e3, 4000e3)
        e    = np.random.uniform(0, 0.35)
        inc  = np.random.uniform(0, np.pi)
        raan = np.random.uniform(0, 2*np.pi)
        argp = np.random.uniform(0, 2*np.pi)
        M0   = np.random.uniform(0, 2*np.pi)
        T    = 2*np.pi*np.sqrt(a**3/mu)
        m    = {"a":a,"e":e,"inc":inc,"raan":raan,"argp":argp,"M0":M0,"body":body}
        times    = np.linspace(0, 3*T, 400)
        positions = propagate_orbit(m, times)
        sp       = sun_position(times, body)
        ef       = eclipse_factor(positions, sp, body)
        events   = extract_events(times, ef, body)
        T_h = T/3600
        feats = [a/1e6, e, np.degrees(inc), np.degrees(raan), np.degrees(argp),
                 np.cos(M0), np.sin(M0), T_h,
                 (a*(1-e)-Rb)/1e3, (a*(1+e)-Rb)/1e3, np.cos(inc), np.sin(inc)]
        durs = [ev["duration"] for ev in events] if events else [0]
        labels = [np.mean(durs), len(events)/3, max(durs)]
        X.append(feats); y.append(labels)

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, random_state=42)),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=42, early_stopping=True),
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        r2s = [r2_score(y_te[:,j], y_pred[:,j]) for j in range(3)]
        maes = [mean_absolute_error(y_te[:,j], y_pred[:,j]) for j in range(3)]
        results[name] = {"pipe": pipe, "r2": r2s, "mae": maes, "mean_r2": np.mean(r2s)}

    best = max(results, key=lambda k: results[k]["mean_r2"])
    return results, best

# ─── Plotting Functions ───────────────────────────────────────────────────────
def plot_orbit_2d(positions, ef, sun_pos, m, color, highlight_frame=None):
    body  = m["body"]
    Rb    = R_BODY[body]
    mu    = MU[body]
    T     = 2 * np.pi * np.sqrt(m["a"]**3 / mu)
    scale = 1e6  # Convert to Mm

    fig = go.Figure()

    # Planet sphere
    theta = np.linspace(0, 2*np.pi, 100)
    px_ = Rb * np.cos(theta) / scale
    py_ = Rb * np.sin(theta) / scale
    body_color = {"Earth":"#2a6099","Mars":"#c0622a","Moon":"#666666"}[body]
    fig.add_trace(go.Scatter(x=px_, y=py_, fill="toself",
        fillcolor=body_color, line=dict(color=body_color, width=1),
        name=body, hoverinfo="skip"))

    # Shadow cone
    sun_dir = sun_pos[0] / np.linalg.norm(sun_pos[0])
    cone_len = m["a"] * 3 / scale
    perp = np.array([-sun_dir[1], sun_dir[0]])
    shadow_w = Rb * 1.1 / scale
    sx, sy = -sun_dir[0], -sun_dir[1]
    fig.add_trace(go.Scatter(
        x=[sx*shadow_w + perp[0]*shadow_w, sx*cone_len, sx*shadow_w - perp[0]*shadow_w],
        y=[sy*shadow_w + perp[1]*shadow_w, sy*cone_len, sy*shadow_w - perp[1]*shadow_w],
        fill="toself", fillcolor="rgba(0,0,0,0.35)",
        line=dict(color="rgba(0,0,0,0)"), name="Shadow", hoverinfo="skip"))

    # Orbit segments — colored by eclipse type
    pos_Mm = positions / scale
    sunlit_x, sunlit_y = [], []
    eclipse_x, eclipse_y = [], []
    penum_x, penum_y = [], []
    for i in range(len(pos_Mm)):
        if ef[i] >= 1:
            eclipse_x.append(pos_Mm[i,0]); eclipse_y.append(pos_Mm[i,1])
        elif ef[i] > 0:
            penum_x.append(pos_Mm[i,0]); penum_y.append(pos_Mm[i,1])
        else:
            sunlit_x.append(pos_Mm[i,0]); sunlit_y.append(pos_Mm[i,1])

    fig.add_trace(go.Scatter(x=sunlit_x, y=sunlit_y, mode="markers",
        marker=dict(size=2, color=color, opacity=0.4), name="Sunlit", hoverinfo="skip"))
    if penum_x:
        fig.add_trace(go.Scatter(x=penum_x, y=penum_y, mode="markers",
            marker=dict(size=3, color="#e3b341"), name="Penumbra", hoverinfo="skip"))
    if eclipse_x:
        fig.add_trace(go.Scatter(x=eclipse_x, y=eclipse_y, mode="markers",
            marker=dict(size=3.5, color="#f85149"), name="Umbra", hoverinfo="skip"))

    # Sun indicator
    sun_edge = m["a"] * 1.4 / scale
    fig.add_trace(go.Scatter(
        x=[sun_dir[0]*sun_edge], y=[sun_dir[1]*sun_edge],
        mode="markers+text",
        marker=dict(size=18, color="#ffd700", symbol="circle",
                    line=dict(color="#ff8c00", width=2)),
        text=["☀ SUN"], textposition="top center",
        textfont=dict(color="#ffd700", size=10, family="Courier New"),
        name="Sun", hoverinfo="skip"))

    # Satellite marker — use highlight_frame for real-time position
    if highlight_frame is not None and highlight_frame < len(ef):
        sat_pos_full = positions  # full orbit already passed in live mode
        # Draw full orbit trail faintly
        all_x = [p[0]/scale for p in sat_pos_full]
        all_y = [p[1]/scale for p in sat_pos_full]
        fig.add_trace(go.Scatter(x=all_x, y=all_y, mode="lines",
            line=dict(color=color, width=0.5), opacity=0.2,
            name="Full Orbit", hoverinfo="skip"))
        sat_idx = highlight_frame
        ef_sat  = ef[sat_idx] if sat_idx < len(ef) else 0
    else:
        sat_idx = len(positions) - 1
        ef_sat  = ef[-1]

    sat = positions[sat_idx] / scale
    sat_color = "#f85149" if ef_sat >= 1 else "#e3b341" if ef_sat > 0 else color
    sat_mode  = "UMBRA" if ef_sat >= 1 else "PENUMBRA" if ef_sat > 0 else "SUNLIT"

    # Pulse ring around satellite
    fig.add_trace(go.Scatter(
        x=[sat[0]], y=[sat[1]], mode="markers",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(color=sat_color, width=1.5)),
        name="", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[sat[0]], y=[sat[1]], mode="markers",
        marker=dict(size=12, color=sat_color, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        name="Satellite",
        hovertemplate=f"<b>Satellite</b><br>Mode: {sat_mode}<extra></extra>"))

    fig.update_layout(
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        font=dict(family="Courier New", color="#8b949e", size=10),
        showlegend=True,
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2333", borderwidth=1,
                    font=dict(size=9, color="#8b949e")),
        xaxis=dict(title="X [Mm]", gridcolor="#1c2333", zeroline=False, color="#445"),
        yaxis=dict(title="Y [Mm]", gridcolor="#1c2333", zeroline=False,
                   scaleanchor="x", color="#445"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=480,
    )
    return fig

def plot_eclipse_timeline(times, ef, events, color):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08,
                        subplot_titles=["Eclipse Factor Over Time", "Eclipse Events (Gantt)"])

    t_min = times / 60
    fig.add_trace(go.Scatter(
        x=t_min, y=ef, fill="tozeroy",
        fillcolor="rgba(248,81,73,0.2)",
        line=dict(color="#f85149", width=1.5),
        name="Eclipse Factor"), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color="#e3b341", dash="dash", width=1),
                  annotation_text="Penumbra", row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color="#f85149", dash="dash", width=1),
                  annotation_text="Umbra", row=1, col=1)

    for ev in events:
        c = "#f85149" if ev["type"]=="umbra" else "#e3b341"
        fig.add_trace(go.Bar(
            x=[ev["duration"]], y=["Eclipse Events"],
            base=[ev["start"]], orientation="h",
            marker_color=c, name=ev["type"].capitalize(),
            hovertemplate=f"<b>{ev['type'].upper()}</b><br>Start: {ev['start']:.1f} min<br>Duration: {ev['duration']:.1f} min<extra></extra>",
            showlegend=False), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        font=dict(family="Courier New", color="#8b949e", size=10),
        showlegend=False,
        xaxis=dict(title="Mission Time [min]", gridcolor="#1c2333", color="#445"),
        yaxis=dict(gridcolor="#1c2333", color="#445"),
        xaxis2=dict(title="Mission Time [min]", gridcolor="#1c2333", color="#445"),
        yaxis2=dict(gridcolor="#1c2333", color="#445"),
        margin=dict(l=60, r=20, t=40, b=40), height=420,
    )
    return fig

def plot_telemetry(times, ef, battery_start):
    t_min = times / 60
    battery = [battery_start]
    solar, draw = [], []
    for i in range(len(times)):
        in_e = ef[i] > 0
        sol  = 0 if in_e else 500
        drw  = 150 if in_e else 300
        dt   = (times[i] - times[i-1]) / 3600 if i > 0 else 0
        bat  = min(100, max(0, battery[-1] + (sol - drw) * dt / 100))
        battery.append(bat); solar.append(sol); draw.append(drw)

    battery = battery[1:]
    fig = make_subplots(rows=3, cols=1, row_heights=[0.4, 0.35, 0.25],
                        vertical_spacing=0.06,
                        subplot_titles=["Battery State of Charge [%]", "Power Budget [W]", "Eclipse Factor"])

    fig.add_trace(go.Scatter(x=t_min, y=battery, fill="tozeroy",
        fillcolor="rgba(63,185,80,0.15)", line=dict(color="#3fb950", width=2),
        name="Battery %"), row=1, col=1)
    fig.add_hline(y=20, line=dict(color="#f85149", dash="dash", width=1.5),
                  annotation_text="Min 20%", row=1, col=1)

    fig.add_trace(go.Scatter(x=t_min, y=solar, fill="tozeroy",
        fillcolor="rgba(227,179,65,0.15)", line=dict(color="#e3b341", width=1.5),
        name="Solar [W]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=draw,
        line=dict(color="#f85149", width=1.5, dash="dash"),
        name="Draw [W]"), row=2, col=1)

    fig.add_trace(go.Scatter(x=t_min, y=ef, fill="tozeroy",
        fillcolor="rgba(248,81,73,0.2)", line=dict(color="#f85149", width=1.5),
        name="Eclipse"), row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        font=dict(family="Courier New", color="#8b949e", size=10),
        showlegend=True,
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2333", borderwidth=1, font=dict(size=9)),
        xaxis3=dict(title="Mission Time [min]", gridcolor="#1c2333", color="#445"),
        margin=dict(l=60, r=20, t=40, b=40), height=520,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1c2333", color="#445", row=i, col=1)
        fig.update_yaxes(gridcolor="#1c2333", color="#445", row=i, col=1)
    return fig

def plot_model_comparison(results):
    labels = ["Mean Eclipse", "Eclipses/Orbit", "Max Eclipse"]
    colors = {"Random Forest":"#00d4ff","Gradient Boosting":"#3fb950","Neural Network":"#a371f7"}
    fig = go.Figure()
    x = np.arange(len(labels))
    w = 0.25
    for i, (name, res) in enumerate(results.items()):
        fig.add_trace(go.Bar(
            x=[l + f" " for l in labels],
            y=res["r2"], name=name,
            marker_color=colors[name],
            text=[f"{v:.3f}" for v in res["r2"]],
            textposition="outside",
            textfont=dict(size=9, color=colors[name]),
        ))
    fig.update_layout(
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        font=dict(family="Courier New", color="#8b949e", size=10),
        barmode="group",
        xaxis=dict(gridcolor="#1c2333", color="#445"),
        yaxis=dict(title="R² Score", gridcolor="#1c2333", color="#445", range=[0, 1.15]),
        legend=dict(bgcolor="#0d1117", bordercolor="#1c2333", borderwidth=1),
        margin=dict(l=60, r=20, t=40, b=40), height=380,
        title=dict(text="AI Model Comparison — R² Score per Target", font=dict(color="#00d4ff", size=13))
    )
    fig.add_hline(y=1.0, line=dict(color="#1c2333", dash="dash", width=1))
    return fig

def plot_feature_importance(best_model_pipe, feature_names, color):
    try:
        model = best_model_pipe.named_steps["model"]
        if hasattr(model, "estimators_"):
            imp = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        else:
            imp = np.ones(len(feature_names)) / len(feature_names)
    except:
        imp = np.ones(len(feature_names)) / len(feature_names)

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
        marker_color=color, text=[f"{v:.1%}" for v in df["importance"]],
        textposition="outside", textfont=dict(size=9, color=color)
    ))
    fig.update_layout(
        paper_bgcolor="#080c12", plot_bgcolor="#080c12",
        font=dict(family="Courier New", color="#8b949e", size=10),
        xaxis=dict(title="Importance", gridcolor="#1c2333", color="#445"),
        yaxis=dict(gridcolor="#1c2333", color="#445"),
        margin=dict(l=140, r=80, t=20, b=40), height=380,
        title=dict(text="Feature Importance (Random Forest)", font=dict(color="#00d4ff", size=12))
    )
    return fig

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <div style='font-family:monospace; font-size:20px; font-weight:bold; color:#00d4ff; letter-spacing:3px;'>
        🛰️ ECLIPSE·PRED
      </div>
      <div style='font-family:monospace; font-size:9px; color:#445; letter-spacing:3px; margin-top:4px;'>
        SATELLITE MISSION CONTROL
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    mission_name = st.selectbox("SELECT MISSION", list(MISSIONS.keys()), index=0)
    m = MISSIONS[mission_name]
    color = m["color"]

    st.divider()
    st.markdown("<div style='font-family:monospace; font-size:9px; color:#445; letter-spacing:2px;'>SIMULATION SETTINGS</div>", unsafe_allow_html=True)
    n_orbits = st.slider("Number of Orbits", 1, 10, 5)
    steps    = st.slider("Time Resolution", 200, 1200, 600, step=100)

    st.divider()
    st.markdown("<div style='font-family:monospace; font-size:9px; color:#445; letter-spacing:2px;'>ORBITAL ELEMENTS</div>", unsafe_allow_html=True)
    mu = MU[m["body"]]
    T  = 2 * np.pi * np.sqrt(m["a"]**3 / mu)
    Rb = R_BODY[m["body"]]
    st.markdown(f"""
    <div style='font-family:monospace; font-size:11px; color:#8b949e; line-height:2;'>
    a = {m['a']/1e6:.3f} Mm<br>
    e = {m['e']:.4f}<br>
    i = {np.degrees(m['inc']):.1f}°<br>
    T = {T/60:.1f} min<br>
    Body = {m['body']}<br>
    Alt = {(m['a']-Rb)/1e3:.0f} km
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    run_btn = st.button("▶ RUN SIMULATION", use_container_width=True, type="primary")

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>
  <div style='width:10px;height:10px;border-radius:50%;background:{color};box-shadow:0 0 8px {color};'></div>
  <div>
    <span style='font-family:monospace; font-size:16px; font-weight:bold; color:{color};'>{mission_name}</span>
    <span style='font-family:monospace; font-size:11px; color:#445; margin-left:12px;'>{m["desc"]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Run Simulation ───────────────────────────────────────────────────────────
if "sim_results" not in st.session_state or run_btn or st.session_state.get("last_mission") != mission_name:
    with st.spinner("Running orbital simulation..."):
        times, positions, sun_pos, ef, T = run_simulation(m, n_orbits, steps)
        events = extract_events(times, ef, m["body"])
        st.session_state.sim_results = (times, positions, sun_pos, ef, T, events)
        st.session_state.last_mission = mission_name

times, positions, sun_pos, ef, T, events = st.session_state.sim_results

# ─── Pre-compute battery for every frame ─────────────────────────────────────
def compute_battery(times, ef, battery_start):
    battery = [battery_start]
    for i in range(1, len(times)):
        in_e = ef[i] > 0
        sol  = 0 if in_e else 500
        drw  = 150 if in_e else 300
        dt   = (times[i] - times[i-1]) / 3600
        bat  = min(100, max(0, battery[-1] + (sol - drw) * dt / 100))
        battery.append(bat)
    return battery

if "battery_series" not in st.session_state or st.session_state.get("last_mission") != mission_name:
    st.session_state.battery_series = compute_battery(times, ef, m["battery_start"])
battery_series = st.session_state.battery_series

# ─── Train AI model ───────────────────────────────────────────────────────────
with st.spinner("Loading AI model (first run takes ~30 seconds)..."):
    model_results, best_model_name = train_model()

best_pipe = model_results[best_model_name]["pipe"]
features  = build_features(m)
pred      = best_pipe.predict(features)[0]

# ─── Stats ────────────────────────────────────────────────────────────────────
durations       = [ev["duration"] for ev in events] if events else [0]
mean_eclipse    = np.mean(durations)
max_eclipse     = np.max(durations)
eclipse_frac    = np.sum(ef > 0) / len(ef) * 100
current_mode    = "UMBRA" if ef[-1] >= 1 else "PENUMBRA" if ef[-1] > 0 else "SUNLIT"
mode_color      = "#f85149" if current_mode=="UMBRA" else "#e3b341" if current_mode=="PENUMBRA" else "#3fb950"

# ─── Live metrics driven by animation frame ───────────────────────────────────
live_frame   = st.session_state.get("anim_frame", 0)
live_ef      = ef[live_frame]
live_battery = battery_series[live_frame]
live_t_min   = times[live_frame] / 60
live_mode    = "UMBRA" if live_ef >= 1 else "PENUMBRA" if live_ef > 0 else "SUNLIT"
live_color   = "#f85149" if live_mode=="UMBRA" else "#e3b341" if live_mode=="PENUMBRA" else "#3fb950"
live_solar   = 0 if live_ef > 0 else 500
live_draw    = 150 if live_ef > 0 else 300
live_net     = live_solar - live_draw
bat_delta    = f"{live_battery - m['battery_start']:+.1f}% from start"
bat_color    = "#f85149" if live_battery < 25 else "#e3b341" if live_battery < 50 else "#3fb950"

# Eclipse countdown — find next eclipse start from current frame
next_eclipse_min = None
for k in range(live_frame, len(ef)):
    if ef[k] > 0 and (live_frame == 0 or ef[live_frame] == 0):
        next_eclipse_min = (times[k] - times[live_frame]) / 60
        break

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("ALTITUDE", f"{(m['a']-Rb)/1e3:.0f} km")
with c2: st.metric("MISSION TIME", f"{live_t_min:.1f} min")
with c3:
    st.markdown(f"""<div data-testid="metric-container" style="background:linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));border:1px solid #1c2333;border-radius:6px;padding:12px 16px;">
    <label style="color:#556;font-family:monospace;font-size:11px;letter-spacing:2px;">BATTERY</label>
    <div style="font-family:monospace;font-size:28px;font-weight:bold;color:{bat_color};">{live_battery:.1f}%</div>
    <div style="font-family:monospace;font-size:11px;color:#445;">{bat_delta}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div data-testid="metric-container" style="background:linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));border:1px solid #1c2333;border-radius:6px;padding:12px 16px;">
    <label style="color:#556;font-family:monospace;font-size:11px;letter-spacing:2px;">SOLAR POWER</label>
    <div style="font-family:monospace;font-size:28px;font-weight:bold;color:{'#e3b341' if live_solar>0 else '#f85149'};">{live_solar}W</div>
    <div style="font-family:monospace;font-size:11px;color:#445;">Net: {live_net:+d}W</div>
    </div>""", unsafe_allow_html=True)
with c5: st.metric("AI PREDICTED", f"{pred[0]:.1f} min", delta=f"{pred[0]-mean_eclipse:+.1f} vs physics")
with c6: st.metric("EVENTS", f"{len(events)}", f"in {n_orbits} orbits")

st.markdown(f"""
<div style='display:flex; align-items:center; gap:8px; margin:8px 0 16px 0;'>
  <span style='font-family:monospace; font-size:11px; color:#445;'>Current mode:</span>
  <span style='font-family:monospace; font-size:12px; font-weight:bold; color:{live_color};
    background:{live_color}18; padding:3px 12px; border-radius:3px; border:{live_color}44 1px solid;'>
    ● {live_mode}
  </span>
  <span style='font-family:monospace;font-size:11px;color:#445;margin-left:8px;'>
    Draw: {live_draw}W
  </span>
  <span style='font-family:monospace; font-size:10px; color:#445; margin-left:8px;'>
    Best AI Model: {best_model_name} · Mean R² = {model_results[best_model_name]['mean_r2']:.3f}
  </span>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⊙  ORBIT VIEW",
    "▦  ECLIPSE MAP",
    "⚡  TELEMETRY",
    "◈  AI MODEL",
    "📋  MISSION DATA"
])

with tab1:
    import streamlit.components.v1 as components

    # Pass orbital data to JS as JSON
    import json

    # Serialize positions, ef, sun_pos, times for JS
    # Downsample to 500 points max for performance
    ds = max(1, len(positions) // 500)
    pos_js  = positions[::ds, :2].tolist()   # only x,y
    ef_js   = ef[::ds].tolist()
    t_js    = times[::ds].tolist()
    bat_js  = battery_series[::ds]
    sun_js  = sun_pos[0, :2].tolist()        # sun direction (fixed for this sim)

    orbit_data = {
        "positions": pos_js,
        "ef": ef_js,
        "times": t_js,
        "battery": bat_js,
        "T": float(T),
        "body": m["body"],
        "a": float(m["a"]),
        "color": color,
        "battery_start": float(m["battery_start"]),
        "sun": sun_js,
        "R_body": float(R_BODY[m["body"]]),
        "mission": mission_name.split("—")[0].strip(),
        "mean_eclipse": float(mean_eclipse),
        "eclipse_frac": float(eclipse_frac),
        "n_events": len(events),
        "ai_pred": float(pred[0]),
        "period_min": float(T/60),
    }

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#080c12; font-family:'Courier New',monospace; color:#8b949e; overflow:hidden; }}
  #wrap {{ display:flex; gap:0; width:100%; height:520px; }}
  #canvasWrap {{ flex:1; position:relative; }}
  canvas {{ display:block; width:100%; height:100%; }}
  #panel {{ width:220px; background:#0d1117; border-left:1px solid #1c2333; padding:16px 12px; display:flex; flex-direction:column; gap:10px; overflow:hidden; }}
  .label {{ font-size:9px; color:#445; letter-spacing:2px; margin-bottom:3px; }}
  .value {{ font-size:16px; font-weight:bold; color:#00d4ff; }}
  .subval {{ font-size:10px; color:#445; margin-top:1px; }}
  .divider {{ border-top:1px solid #1c2333; margin:2px 0; }}
  .mode-badge {{ font-size:11px; font-weight:bold; padding:4px 10px; border-radius:3px; display:inline-block; margin-top:2px; }}
  .bat-bar-wrap {{ height:10px; background:#1c2333; border-radius:5px; overflow:hidden; margin-top:4px; }}
  .bat-bar {{ height:100%; border-radius:5px; transition:width 0.5s, background 0.5s; }}
  #clock {{ font-size:10px; color:#445; margin-top:2px; }}
  #nextEclipse {{ font-size:11px; color:#e3b341; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="canvasWrap"><canvas id="c"></canvas></div>
  <div id="panel">
    <div>
      <div class="label">MISSION</div>
      <div class="value" style="font-size:13px;" id="missionName"></div>
    </div>
    <div class="divider"></div>
    <div>
      <div class="label">UTC TIME</div>
      <div id="clock">--</div>
    </div>
    <div>
      <div class="label">ORBIT POSITION</div>
      <div class="value" id="tVal">--</div>
      <div class="subval" id="orbitNum">--</div>
    </div>
    <div class="divider"></div>
    <div>
      <div class="label">CURRENT MODE</div>
      <div id="modeBadge" class="mode-badge">--</div>
    </div>
    <div>
      <div class="label">NEXT ECLIPSE</div>
      <div id="nextEclipse">--</div>
    </div>
    <div class="divider"></div>
    <div>
      <div class="label">BATTERY</div>
      <div class="value" id="batVal">--</div>
      <div class="bat-bar-wrap"><div class="bat-bar" id="batBar"></div></div>
    </div>
    <div>
      <div class="label">SOLAR POWER</div>
      <div class="value" id="solarVal">--</div>
      <div class="subval" id="netPower">--</div>
    </div>
    <div class="divider"></div>
    <div>
      <div class="label">ECLIPSE STATS</div>
      <div class="subval">Mean: <span id="meanEcl" style="color:#00d4ff;"></span> min</div>
      <div class="subval">Fraction: <span id="eclFrac" style="color:#00d4ff;"></span>%</div>
      <div class="subval">Events: <span id="nEvents" style="color:#00d4ff;"></span></div>
      <div class="subval">AI pred: <span id="aiPred" style="color:#a371f7;"></span> min</div>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(orbit_data)};

// ── Pre-process ────────────────────────────────────────────────────
const pos   = DATA.positions;
const ef    = DATA.ef;
const times = DATA.times;
const bat   = DATA.battery;
const T     = DATA.T;          // orbital period in seconds
const color = DATA.color;
const Rb    = DATA.R_body;
const sunDir = DATA.sun;
const N     = pos.length;

// Normalise all positions for canvas
const maxR  = Math.max(...pos.map(p => Math.sqrt(p[0]*p[0]+p[1]*p[1])));

// ── Canvas setup ──────────────────────────────────────────────────
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

function resize() {{
  const wrap = document.getElementById('canvasWrap');
  canvas.width  = wrap.clientWidth  * window.devicePixelRatio;
  canvas.height = wrap.clientHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}}
window.addEventListener('resize', resize);
resize();

function cx() {{ return canvas.width  / window.devicePixelRatio / 2; }}
function cy() {{ return canvas.height / window.devicePixelRatio / 2; }}
function scale() {{ return Math.min(cx(), cy()) * 0.82; }}

function toScreen(x, y) {{
  const s = scale() / maxR;
  return [cx() + x * s, cy() - y * s];
}}

// ── J2000 epoch ───────────────────────────────────────────────────
const J2000 = Date.UTC(2000, 0, 1, 12, 0, 0);

function getCurrentFrame() {{
  const now    = Date.now();
  const elapsed = (now - J2000) / 1000;   // seconds since J2000
  const orbit_t = elapsed % T;             // position within current orbit (seconds)
  // Binary search for closest frame
  let lo = 0, hi = N - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (times[mid] < orbit_t) lo = mid + 1;
    else hi = mid;
  }}
  return Math.min(lo, N - 1);
}}

// ── Battery simulation — run in real time ────────────────────────
// Recompute battery from frame 0 up to current frame each tick
// (fast since N<=500)
function getBattery(frameIdx) {{
  return bat[frameIdx];
}}

// ── Draw ──────────────────────────────────────────────────────────
function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `${{r}},${{g}},${{b}}`;
}}

function draw() {{
  const W = canvas.width  / window.devicePixelRatio;
  const H = canvas.height / window.devicePixelRatio;
  ctx.clearRect(0, 0, W, H);

  const f = getCurrentFrame();

  // ── Background stars ─────────────────────────────────────────
  ctx.fillStyle = '#080c12';
  ctx.fillRect(0, 0, W, H);
  // Static stars (seeded)
  const starSeed = 42;
  for (let i = 0; i < 120; i++) {{
    const sx = ((Math.sin(i * 127.1 + starSeed) * 0.5 + 0.5)) * W;
    const sy = ((Math.sin(i * 311.7 + starSeed) * 0.5 + 0.5)) * H;
    const sr = 0.5 + (Math.sin(i * 74.3) * 0.5 + 0.5) * 1.0;
    const sa = 0.3 + (Math.sin(i * 53.1) * 0.5 + 0.5) * 0.5;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${{sa}})`;
    ctx.fill();
  }}

  // ── Sun direction & rays ──────────────────────────────────────
  const sunMag  = Math.sqrt(sunDir[0]**2 + sunDir[1]**2);
  const sunNorm = [sunDir[0]/sunMag, sunDir[1]/sunMag];
  const sunDist = scale() * 1.35;
  const [sx, sy] = [cx() + sunNorm[0]*sunDist, cy() - sunNorm[1]*sunDist];

  // Sun glow
  const grd = ctx.createRadialGradient(sx, sy, 2, sx, sy, 40);
  grd.addColorStop(0, 'rgba(255,220,50,0.9)');
  grd.addColorStop(0.3, 'rgba(255,180,30,0.4)');
  grd.addColorStop(1, 'rgba(255,150,0,0)');
  ctx.beginPath(); ctx.arc(sx, sy, 40, 0, Math.PI*2);
  ctx.fillStyle = grd; ctx.fill();
  ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI*2);
  ctx.fillStyle = '#ffd700'; ctx.fill();

  // Shadow cone
  const shadowLen = scale() * 2.2;
  const perpX =  sunNorm[1];
  const perpY = -sunNorm[0];
  const shadowW = (Rb / maxR) * scale() * 1.1;
  const [ox, oy] = toScreen(0, 0);
  ctx.beginPath();
  ctx.moveTo(ox + perpX*shadowW, oy + perpY*shadowW);
  ctx.lineTo(ox - sunNorm[0]*shadowLen, oy + sunNorm[1]*shadowLen);
  ctx.lineTo(ox - perpX*shadowW, oy - perpY*shadowW);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.fill();

  // ── Planet ────────────────────────────────────────────────────
  const pR = (Rb / maxR) * scale();
  const bodyColors = {{Earth:'#2a6099', Mars:'#c0622a', Moon:'#555'}};
  const bColor = bodyColors[DATA.body] || '#2a6099';
  const grad = ctx.createRadialGradient(cx()-pR*0.3, cy()-pR*0.3, pR*0.1, cx(), cy(), pR);
  grad.addColorStop(0, bColor);
  grad.addColorStop(1, '#0a0f18');
  ctx.beginPath(); ctx.arc(cx(), cy(), pR, 0, Math.PI*2);
  ctx.fillStyle = grad; ctx.fill();
  ctx.strokeStyle = bColor + '66'; ctx.lineWidth = 1;
  ctx.stroke();

  // ── Full orbit trail (faint) ──────────────────────────────────
  ctx.beginPath();
  for (let i = 0; i < N; i++) {{
    const [px, py] = toScreen(pos[i][0], pos[i][1]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }}
  ctx.strokeStyle = `rgba(${{hexToRgb(color)}},0.12)`;
  ctx.lineWidth = 1; ctx.stroke();

  // ── Coloured orbit arc up to current frame ────────────────────
  for (let i = 1; i <= f; i++) {{
    const [x1, y1] = toScreen(pos[i-1][0], pos[i-1][1]);
    const [x2, y2] = toScreen(pos[i][0],   pos[i][1]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    if      (ef[i] >= 1)  ctx.strokeStyle = 'rgba(248,81,73,0.7)';
    else if (ef[i] > 0)   ctx.strokeStyle = 'rgba(227,179,65,0.7)';
    else                  ctx.strokeStyle = `rgba(${{hexToRgb(color)}},0.55)`;
    ctx.lineWidth = 1.8; ctx.stroke();
  }}

  // ── Satellite ─────────────────────────────────────────────────
  const [satX, satY] = toScreen(pos[f][0], pos[f][1]);
  const inEclipse = ef[f] >= 1;
  const inPenum   = ef[f] > 0 && ef[f] < 1;
  const satCol    = inEclipse ? '#f85149' : inPenum ? '#e3b341' : color;

  // Pulse ring
  const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 400);
  ctx.beginPath(); ctx.arc(satX, satY, 10 + pulse*5, 0, Math.PI*2);
  ctx.strokeStyle = satCol + '55'; ctx.lineWidth = 1.5; ctx.stroke();

  // Satellite body
  ctx.save();
  ctx.translate(satX, satY);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = satCol;
  ctx.fillRect(-5, -5, 10, 10);
  // Solar panels
  ctx.fillStyle = inEclipse ? '#333' : '#4a9eff';
  ctx.fillRect(-14, -2, 8, 4);
  ctx.fillRect(6, -2, 8, 4);
  ctx.restore();

  // ── Update panel ──────────────────────────────────────────────
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toUTCString().replace('GMT','UTC');
  document.getElementById('missionName').textContent = DATA.mission;

  const tMin   = times[f] / 60;
  const tTotal = T / 60;
  const orbitN = Math.floor(tMin / tTotal) + 1;
  document.getElementById('tVal').textContent =
    tMin.toFixed(1) + ' / ' + tTotal.toFixed(1) + ' min';
  document.getElementById('orbitNum').textContent = 'Orbit ' + orbitN;

  const mode      = inEclipse ? 'UMBRA' : inPenum ? 'PENUMBRA' : 'SUNLIT';
  const modeColor = inEclipse ? '#f85149' : inPenum ? '#e3b341' : '#3fb950';
  const badge     = document.getElementById('modeBadge');
  badge.textContent = '● ' + mode;
  badge.style.color      = modeColor;
  badge.style.background = modeColor + '18';
  badge.style.border     = '1px solid ' + modeColor + '44';

  // Next eclipse countdown
  let nextStr = '—';
  if (!inEclipse && !inPenum) {{
    for (let k = f; k < N; k++) {{
      if (ef[k] > 0) {{
        const mins = (times[k] - times[f]) / 60;
        nextStr = 'in ' + mins.toFixed(1) + ' min';
        break;
      }}
    }}
  }} else {{
    for (let k = f; k < N; k++) {{
      if (ef[k] === 0) {{
        const mins = (times[k] - times[f]) / 60;
        nextStr = 'exits in ' + mins.toFixed(1) + ' min';
        break;
      }}
    }}
  }}
  document.getElementById('nextEclipse').textContent = nextStr;

  // Battery
  const b      = getBattery(f);
  const bColor2 = b < 25 ? '#f85149' : b < 50 ? '#e3b341' : '#3fb950';
  document.getElementById('batVal').textContent = b.toFixed(1) + '%';
  document.getElementById('batBar').style.width      = b + '%';
  document.getElementById('batBar').style.background = bColor2;

  // Solar
  const solar = inEclipse || inPenum ? 0 : 500;
  const draw2  = inEclipse || inPenum ? 150 : 300;
  const net    = solar - draw2;
  document.getElementById('solarVal').textContent = solar + 'W';
  document.getElementById('solarVal').style.color = solar > 0 ? '#e3b341' : '#f85149';
  document.getElementById('netPower').textContent = 'Net: ' + (net >= 0 ? '+' : '') + net + 'W';

  // Static stats
  document.getElementById('meanEcl').textContent = DATA.mean_eclipse.toFixed(1);
  document.getElementById('eclFrac').textContent  = DATA.eclipse_frac.toFixed(1);
  document.getElementById('nEvents').textContent  = DATA.n_events;
  document.getElementById('aiPred').textContent   = DATA.ai_pred.toFixed(1);

  requestAnimationFrame(draw);
}}

draw();
</script>
</body>
</html>
"""

    components.html(html_code, height=520, scrolling=False)

with tab2:
    st.plotly_chart(plot_eclipse_timeline(times, ef, events, color),
                    use_container_width=True, config={"displayModeBar": False})

    if events:
        st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#445;letter-spacing:2px;margin:12px 0 8px 0;'>ECLIPSE EVENT LOG</div>", unsafe_allow_html=True)
        df = pd.DataFrame(events)
        df["start"] = df["start"].round(2)
        df["end"]   = df["end"].round(2)
        df["duration"] = df["duration"].round(2)
        df.columns = ["Start (min)", "End (min)", "Duration (min)", "Type", "Body"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="info-box">No eclipse events detected in this simulation window. Try increasing the number of orbits or adjusting the RAAN angle.</div>', unsafe_allow_html=True)

with tab3:
    # Show telemetry up to current animation frame
    f3 = st.session_state.get("anim_frame", len(times)-1)
    f3 = max(1, f3)
    st.plotly_chart(plot_telemetry(times[:f3], ef[:f3], m["battery_start"]),
                    use_container_width=True, config={"displayModeBar": False})
    # Live battery gauge
    b3 = battery_series[f3-1]
    b3_color = "#f85149" if b3 < 25 else "#e3b341" if b3 < 50 else "#3fb950"
    st.markdown(f"""
    <div style='display:flex;gap:16px;align-items:center;margin-bottom:8px;'>
      <div style='font-family:monospace;font-size:11px;color:#445;'>LIVE BATTERY:</div>
      <div style='flex:1;height:18px;background:#1c2333;border-radius:9px;overflow:hidden;'>
        <div style='width:{b3:.1f}%;height:100%;background:{b3_color};border-radius:9px;
          transition:width 0.3s;'></div>
      </div>
      <div style='font-family:monospace;font-size:14px;font-weight:bold;color:{b3_color};min-width:52px;'>{b3:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="sunlit-box">☀ SUNLIT MODE<br>Solar: 500W · Draw: 300W<br>Net: +200W charging<br>All instruments ON</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="eclipse-box">● ECLIPSE MODE<br>Solar: 0W · Draw: 150W<br>Net: -150W battery drain<br>Instruments OFF · Heaters ON</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-box">◑ RECOVERY MODE<br>Solar: 500W · Draw: 180W<br>Net: +320W fast recharge<br>Gradual instrument restart</div>', unsafe_allow_html=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_model_comparison(model_results),
                        use_container_width=True, config={"displayModeBar": False})
    with col2:
        feature_names = ["Semi-major axis","Eccentricity","Inclination","RAAN",
                         "Arg. Perigee","cos(M0)","sin(M0)","Period",
                         "Periapsis alt","Apoapsis alt","cos(i)","sin(i)"]
        st.plotly_chart(plot_feature_importance(best_pipe, feature_names, color),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#445;letter-spacing:2px;margin:12px 0 8px 0;'>DETAILED MODEL METRICS</div>", unsafe_allow_html=True)
    labels = ["Mean Eclipse (min)", "Eclipses/Orbit", "Max Eclipse (min)"]
    rows = []
    for name, res in model_results.items():
        for j, label in enumerate(labels):
            rows.append({"Model": name, "Target": label,
                         "R²": round(res["r2"][j], 4),
                         "MAE": round(res["mae"][j], 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#445;letter-spacing:2px;margin-bottom:10px;'>MISSION TIMELINE</div>", unsafe_allow_html=True)
        prev_end = 0
        for ev in events[:8]:
            if ev["start"] > prev_end:
                dur = ev["start"] - prev_end
                st.markdown(f'<div class="sunlit-box">☀ SUNLIT · {prev_end:.1f}m → {ev["start"]:.1f}m · {dur:.1f} min<br><small>Science ops · Data TX · Battery charge</small></div>', unsafe_allow_html=True)
            c = "eclipse-box" if ev["type"]=="umbra" else "info-box"
            st.markdown(f'<div class="{c}">{"●" if ev["type"]=="umbra" else "◑"} {ev["type"].upper()} · {ev["start"]:.1f}m → {ev["end"]:.1f}m · {ev["duration"]:.1f} min<br><small>Battery mode · Heaters ON · Instruments OFF</small></div>', unsafe_allow_html=True)
            prev_end = ev["end"]

    with c2:
        st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#445;letter-spacing:2px;margin-bottom:10px;'>ALL MISSIONS COMPARISON</div>", unsafe_allow_html=True)
        comp_data = []
        for mname, mm in MISSIONS.items():
            mu2 = MU[mm["body"]]
            T2  = 2*np.pi*np.sqrt(mm["a"]**3/mu2)
            Rb2 = R_BODY[mm["body"]]
            t2  = np.linspace(0, 3*T2, 400)
            p2  = propagate_orbit(mm, t2)
            s2  = sun_position(t2, mm["body"])
            e2  = eclipse_factor(p2, s2, mm["body"])
            ev2 = extract_events(t2, e2, mm["body"])
            dur2 = [ev["duration"] for ev in ev2] if ev2 else [0]
            comp_data.append({
                "Mission": mname.split("—")[0].strip(),
                "Body": mm["body"],
                "Altitude (km)": int((mm["a"]-Rb2)/1e3),
                "Period (min)": round(T2/60, 1),
                "Eclipse Events": len(ev2),
                "Mean Eclipse (min)": round(np.mean(dur2), 1),
                "Eclipse %": round(np.sum(e2>0)/len(e2)*100, 1),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
