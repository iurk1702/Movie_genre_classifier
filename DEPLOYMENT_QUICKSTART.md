# Quick Deployment Guide

## 🚀 Quick Start

### Backend (Render) - 5 Steps

1. **Commit model files** (if not already):
   ```bash
   cd Movie_genre_classifier
   git add -f models/*.pkl models/metadata.json data/research/*.json
   git commit -m "Add model files for deployment"
   git push
   ```

2. **Go to Render**: https://dashboard.render.com → "New +" → "Web Service"

3. **Connect GitHub repo**: Select `Movie_genre_classifier`

4. **Configure**:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Env vars: `ENVIRONMENT=production`, `DEBUG=false`

5. **Deploy** → Copy backend URL (e.g., `https://xxx.onrender.com`)

### Frontend (Vercel) - 4 Steps

1. **Go to Vercel**: https://vercel.com/dashboard → "Add New" → "Project"

2. **Import repo**: Select `genre-dialogue-lab`

3. **Add env var**:
   - Key: `VITE_API_BASE_URL`
   - Value: `https://your-backend.onrender.com/api`

4. **Deploy** → Copy frontend URL

### Final Step: Update CORS

In Render dashboard, add env var:
- Key: `FRONTEND_URL`
- Value: `https://your-frontend.vercel.app`

## ✅ Checklist

- [ ] Backend deployed on Render
- [ ] Backend URL copied
- [ ] Frontend deployed on Vercel
- [ ] `VITE_API_BASE_URL` set in Vercel
- [ ] `FRONTEND_URL` set in Render
- [ ] Both apps tested and working

## 📚 Full Guide

See `DEPLOYMENT.md` for detailed instructions and troubleshooting.

