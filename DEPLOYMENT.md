# 🚀 Deployment Guide - QML-Liver Disease Detection

This platform supports **100% free cloud deployment** across multiple leading platforms.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended - Easiest & 100% Free Forever)

Official cloud platform tailored for Streamlit apps with zero config and zero maintenance.

### Steps:
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account (`AYUSH1605915`).
2. Click **"New app"**.
3. Fill in the deployment details:
   - **Repository**: `AYUSH1605915/quantum-machine-learning-`
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
   - **App URL**: Choose your custom URL (e.g. `qml-liver.streamlit.app`)
4. Click **"Deploy!"**
5. Within 1–2 minutes, your live public URL will be active and shareable worldwide with automatic SSL (HTTPS).

---

## ⚡ Option 2: Render.com (100% Free Web Service)

Render provides a free web service tier that natively reads the included `render.yaml` or `Procfile`.

### Steps:
1. Go to **[render.com](https://dashboard.render.com/)** and sign in with GitHub.
2. Click **"New +"** → **"Web Service"**.
3. Connect your GitHub repository: `AYUSH1605915/quantum-machine-learning-`.
4. Configure:
   - **Name**: `qml-liver-detection`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance Type**: `Free`
5. Click **"Deploy Web Service"**.
6. Your live public URL will be ready at `https://qml-liver-detection.onrender.com`.

---

## 🤗 Option 3: Hugging Face Spaces (Free Forever, AI/ML Native)

Hugging Face Spaces is designed specifically for machine learning and quantum models with unlimited free runtime.

### Steps:
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Set:
   - **Space name**: `quantum-liver-detection`
   - **License**: `mit`
   - **Space SDK**: Select **Streamlit** (or **Docker**).
3. Connect your GitHub repository or push this repository directly to your HF Space Git remote.
4. Set entrypoint to `app/dashboard.py`.
5. Your space will build and launch at `https://huggingface.co/spaces/AYUSH1605915/quantum-liver-detection`.

---

## ▲ Option 4: Vercel (Free Serverless Deployment)

The repository includes `vercel.json` for deploying the Flask API platform (`app.py`).

### Steps:
1. Go to **[vercel.com](https://vercel.com/)** and login with GitHub.
2. Click **"Add New..."** → **"Project"**.
3. Select `quantum-machine-learning-` from your repository list.
4. Framework Preset: Select **Other**.
5. Click **"Deploy"**.
6. Vercel will deploy the application on `https://quantum-machine-learning-*.vercel.app`.
