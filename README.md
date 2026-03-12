# 🛰️ Eclipse·Pred — AI Satellite Eclipse Prediction System

An AI-assisted satellite eclipse prediction system for deep-space missions.
Built with Streamlit, Plotly, and scikit-learn.

## 🚀 Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
# Create a new GitHub repo called "satellite-eclipse"
# Then in your terminal:

cd streamlit_app
git init
git add .
git commit -m "Initial commit — Eclipse Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/satellite-eclipse.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo: `satellite-eclipse`
5. Branch: `main`
6. Main file: `app.py`
7. Click **"Deploy!"**

✅ Your app will be live at:
`https://YOUR_USERNAME-satellite-eclipse-app-XXXXX.streamlit.app`

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at: http://localhost:8501

---

## 📁 Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🌟 Features

- **4 real missions**: ISS, GPS, MRO (Mars), LRO (Moon)
- **Live orbital simulation** using Kepler's equation
- **Eclipse detection** with umbra/penumbra geometry
- **AI prediction** with Random Forest, Gradient Boosting, Neural Network
- **Power telemetry** — battery, solar panels, autonomous modes
- **Interactive Plotly charts** — all visualizations
- **Mission comparison table** — all 4 satellites side by side
