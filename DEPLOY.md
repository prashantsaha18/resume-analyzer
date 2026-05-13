# ResumeAI Pro — Vercel Deployment Guide

Two separate Vercel projects: one for the **frontend**, one for the **backend**.

---

## Prerequisites
- Vercel account at vercel.com (free)
- Neon database configured (you already have this)
- Firebase project configured (you already have this)

---

## Step 1 — Deploy the Backend

### 1a. Push to GitHub
```bash
cd resumeai-pro
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/resumeai-pro.git
git push -u origin main
```

### 1b. Create Backend Project on Vercel
1. Go to vercel.com → **Add New Project**
2. Import your GitHub repo
3. **IMPORTANT**: Set Root Directory → `backend`
4. Framework Preset → **Other**
5. Build Command → leave blank (vercel.json handles it)
6. Click **Deploy**

### 1c. Add Backend Environment Variables
In Vercel Dashboard → your backend project → **Settings → Environment Variables**:

Add each of these:

| Key | Value |
|-----|-------|
| `DJANGO_SETTINGS_MODULE` | `resumeai.settings_vercel` |
| `DJANGO_SECRET_KEY` | _(run: `python -c "import secrets; print(secrets.token_urlsafe(50))"`)_ |
| `DB_NAME` | your Neon DB name |
| `DB_USER` | your Neon username |
| `DB_PASSWORD` | your Neon password |
| `DB_HOST` | your Neon host (ends with .neon.tech) |
| `DB_PORT` | `5432` |
| `FIREBASE_CREDENTIALS_JSON` | _(paste entire firebase JSON as one line — see below)_ |
| `CORS_ALLOWED_ORIGINS` | `https://your-frontend.vercel.app` _(fill after frontend deploys)_ |
| `GEMINI_API_KEY` | _(optional — enables Gemini AI)_ |

### 1d. How to get FIREBASE_CREDENTIALS_JSON
1. Firebase Console → Project Settings → Service Accounts
2. Generate new private key → download JSON
3. Open the JSON file, **copy the entire contents**
4. Paste as the value for `FIREBASE_CREDENTIALS_JSON` in Vercel

### 1e. Redeploy after adding env vars
Vercel Dashboard → your backend → **Deployments** → three dots → **Redeploy**

Your backend URL will be: `https://resumeai-pro-backend.vercel.app`

---

## Step 2 — Deploy the Frontend

### 2a. Create Frontend Project on Vercel
1. Vercel → **Add New Project** → same GitHub repo
2. **IMPORTANT**: Set Root Directory → `frontend`
3. Framework Preset → **Create React App**
4. Click **Deploy**

### 2b. Add Frontend Environment Variables
In Vercel → frontend project → **Settings → Environment Variables**:

| Key | Value |
|-----|-------|
| `REACT_APP_API_URL` | `https://your-backend.vercel.app/api` |
| `REACT_APP_FIREBASE_API_KEY` | from Firebase Console |
| `REACT_APP_FIREBASE_AUTH_DOMAIN` | `your-project.firebaseapp.com` |
| `REACT_APP_FIREBASE_PROJECT_ID` | your project ID |
| `REACT_APP_FIREBASE_STORAGE_BUCKET` | `your-project.appspot.com` |
| `REACT_APP_FIREBASE_MESSAGING_SENDER_ID` | from Firebase Console |
| `REACT_APP_FIREBASE_APP_ID` | from Firebase Console |

### 2c. Add Firebase Authorized Domain
1. Firebase Console → Authentication → Settings → Authorized domains
2. Add your frontend Vercel URL: `your-frontend.vercel.app`

### 2d. Update Backend CORS
Go back to backend Vercel → Settings → Environment Variables
Update `CORS_ALLOWED_ORIGINS` with your actual frontend URL:
```
https://your-frontend.vercel.app
```
Then redeploy the backend.

---

## Step 3 — Verify Deployment

1. Open your frontend URL — you should see the login page
2. Register with an email — Firebase auth should work
3. Go to Dashboard — API calls should succeed
4. Test the Analyzer — AI features should work (rule-based or Gemini)

---

## AI Backend on Vercel

Since Vercel is serverless, scikit-learn ML models can't be stored.
The app automatically uses:
- **Gemini API** if `GEMINI_API_KEY` is set → full AI features
- **Rule-based engine** if no key → still fully functional, no ML

To enable Gemini (free, no credit card):
1. Get key at: https://aistudio.google.com/app/apikey
2. Add to Vercel backend env vars as `GEMINI_API_KEY`
3. Redeploy backend

---

## Custom Domain (optional)

1. Vercel → your project → Settings → Domains
2. Add your domain (e.g. `app.resumeai.pro`)
3. Follow DNS instructions Vercel provides
4. Update `CORS_ALLOWED_ORIGINS` in backend with new domain
5. Update `REACT_APP_API_URL` in frontend with backend domain

---

## Troubleshooting

**Login fails** → Firebase auth domain not whitelisted (Step 2c)

**API calls fail (CORS error)** → CORS_ALLOWED_ORIGINS mismatch (Step 2d)

**500 errors** → Check Vercel Functions logs: Dashboard → Deployments → Functions tab

**DB connection error** → Verify Neon credentials and that SSL is enabled

**Firebase JSON invalid** → Make sure you pasted the complete JSON with no line breaks