# Deployment Guide

This guide covers deploying both the backend (FastAPI) and frontend (React) applications.

## Overview

- **Backend**: Deploy to [Render.com](https://render.com) (Free tier available)
- **Frontend**: Deploy to [Vercel](https://vercel.com) (Free tier available)

## Prerequisites

1. GitHub accounts with both repositories pushed
2. Render.com account (sign up at https://render.com)
3. Vercel account (sign up at https://vercel.com)
4. Trained model files in the `models/` directory
5. Research data files in `data/research/` directory

---

## Part 1: Backend Deployment on Render

### Step 1: Prepare Your Repository

Ensure your backend repository has:
- ✅ `render.yaml` file (already included)
- ✅ `requirements.txt` with all dependencies
- ✅ Trained model files (`models/*.pkl` and `models/metadata.json`)
- ✅ Research data files (`data/research/*.json`)

**Important**: Model files and research data are gitignored. You need to commit them for deployment:

```bash
cd /Users/vaarunaykaushal/Documents/iurk1702/Movie_genre_classifier

# Temporarily remove from .gitignore (or use git add -f)
git add -f models/*.pkl models/metadata.json
git add -f data/research/*.json

# Commit
git commit -m "feat: Add model files and research data for deployment"
git push origin main
```

**Note**: If model files are too large (>100MB), consider using Git LFS or storing them externally.

### Step 2: Create Render Web Service

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → **"Web Service"**
3. **Connect your GitHub repository**:
   - Select your `Movie_genre_classifier` repository
   - Click "Connect"
4. **Configure the service**:
   - **Name**: `movie-genre-classifier-api` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: Leave empty (root)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Environment Variables** (click "Advanced"):
   - `ENVIRONMENT` = `production`
   - `DEBUG` = `false`
6. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- Render will automatically:
  - Clone your repository
  - Install dependencies
  - Start your application
- First deployment may take 5-10 minutes
- You'll see build logs in real-time

### Step 4: Get Your Backend URL

Once deployed, Render will provide a URL like:
```
https://movie-genre-classifier-api.onrender.com
```

**Important**: Note this URL - you'll need it for frontend deployment.

### Step 5: Test Your Backend

Visit your backend URL:
- Health check: `https://your-backend.onrender.com/health`
- API docs: `https://your-backend.onrender.com/docs`

---

## Part 2: Frontend Deployment on Vercel

### Step 1: Prepare Your Repository

Ensure your frontend repository has:
- ✅ All source files
- ✅ `package.json` with build scripts
- ✅ No `.env` file committed (should be gitignored)

### Step 2: Create Vercel Project

1. **Go to Vercel Dashboard**: https://vercel.com/dashboard
2. **Click "Add New..."** → **"Project"**
3. **Import your GitHub repository**:
   - Select your `genre-dialogue-lab` repository
   - Click "Import"
4. **Configure the project**:
   - **Framework Preset**: Vite (should auto-detect)
   - **Root Directory**: `./` (root)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `dist` (default)
   - **Install Command**: `npm install` (default)

### Step 3: Add Environment Variables

Before deploying, add environment variable:

1. **Click "Environment Variables"** section
2. **Add variable**:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://your-backend.onrender.com/api`
     - Replace `your-backend.onrender.com` with your actual Render backend URL
   - **Environment**: Select all (Production, Preview, Development)
3. **Click "Add"**

### Step 4: Deploy

1. **Click "Deploy"**
2. Vercel will:
   - Install dependencies
   - Build your application
   - Deploy to production
3. First deployment takes 2-5 minutes

### Step 5: Get Your Frontend URL

Once deployed, Vercel will provide a URL like:
```
https://genre-dialogue-lab.vercel.app
```

---

## Part 3: Update Backend CORS Settings

After frontend deployment, update backend CORS to allow your Vercel domain:

### Option A: Update via Render Dashboard

1. Go to your Render service dashboard
2. Click "Environment" tab
3. Add environment variable:
   - **Key**: `FRONTEND_URL`
   - **Value**: `https://your-frontend.vercel.app`
4. Redeploy the service

### Option B: Update Code and Redeploy

1. **Update `app/config.py`**:
   ```python
   CORS_ORIGINS = [
       "http://localhost:3000",
       "http://localhost:5173",
       "http://localhost:5174",
       "http://localhost:8080",
       "https://localhost:5173",
       "https://localhost:5174",
       "https://your-frontend.vercel.app",  # Add your Vercel URL
   ]
   ```

2. **Commit and push**:
   ```bash
   git add app/config.py
   git commit -m "chore: Add Vercel frontend URL to CORS origins"
   git push origin main
   ```

3. **Render will auto-deploy** the changes

---

## Part 4: Verify Deployment

### Test Backend

1. Visit: `https://your-backend.onrender.com/docs`
2. Test the `/health` endpoint
3. Test the `/api/predict` endpoint with sample dialogue

### Test Frontend

1. Visit: `https://your-frontend.vercel.app`
2. Try predicting a genre
3. Check the Research tab loads data

### Common Issues

**CORS Errors**:
- Ensure frontend URL is added to backend CORS origins
- Check environment variable `VITE_API_BASE_URL` is set correctly

**404 Errors on API**:
- Verify `VITE_API_BASE_URL` includes `/api` suffix
- Check backend is running (visit `/health` endpoint)

**Model Not Found**:
- Ensure model files are committed to repository
- Check file paths in `app/config.py` are correct

**Build Failures**:
- Check build logs for specific errors
- Verify all dependencies are in `requirements.txt` (backend) or `package.json` (frontend)

---

## Part 5: Custom Domain (Optional)

### Backend Custom Domain (Render)

1. Go to Render service dashboard
2. Click "Settings" → "Custom Domains"
3. Add your domain
4. Follow DNS configuration instructions

### Frontend Custom Domain (Vercel)

1. Go to Vercel project dashboard
2. Click "Settings" → "Domains"
3. Add your domain
4. Follow DNS configuration instructions
5. Update `VITE_API_BASE_URL` environment variable if needed

---

## Monitoring and Updates

### Render (Backend)

- **Logs**: Available in Render dashboard
- **Auto-deploy**: Enabled by default on `main` branch push
- **Free tier**: Services sleep after 15 minutes of inactivity (first request may be slow)

### Vercel (Frontend)

- **Logs**: Available in Vercel dashboard
- **Auto-deploy**: Enabled by default on `main` branch push
- **Analytics**: Available in Vercel dashboard

### Updating Your Application

1. Make changes locally
2. Commit and push to GitHub
3. Both Render and Vercel will automatically deploy updates

---

## Cost

- **Render Free Tier**: 
  - Web services sleep after inactivity
  - Suitable for development/demos
  - Upgrade for always-on service ($7/month)
  
- **Vercel Free Tier**:
  - Unlimited deployments
  - Perfect for frontend hosting
  - Generous bandwidth limits

---

## Troubleshooting

### Backend Issues

**Service won't start**:
- Check build logs for Python errors
- Verify `requirements.txt` is correct
- Ensure `app/main.py` exists

**Model loading errors**:
- Verify model files are in repository
- Check file paths in `app/config.py`
- Ensure files are not corrupted

### Frontend Issues

**Build fails**:
- Check Node.js version (Vercel auto-detects)
- Verify all dependencies in `package.json`
- Check for TypeScript errors

**API calls fail**:
- Verify `VITE_API_BASE_URL` environment variable
- Check browser console for CORS errors
- Verify backend is running

---

## Quick Reference

### Backend URL
```
https://your-backend.onrender.com
```

### Frontend URL
```
https://your-frontend.vercel.app
```

### Environment Variables

**Backend (Render)**:
- `ENVIRONMENT` = `production`
- `DEBUG` = `false`

**Frontend (Vercel)**:
- `VITE_API_BASE_URL` = `https://your-backend.onrender.com/api`

---

## Support

- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Vite Docs**: https://vitejs.dev

