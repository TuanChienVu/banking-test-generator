# 🚀 Banking Test Case Generator - Deployment Guide

## 📋 Lựa chọn Deployment

### Option 1: Railway (Khuyến nghị - GitHub Student Pack)
**Ưu điểm:**
- ✅ $5 credit/tháng miễn phí với GitHub Student Pack
- ✅ Deploy nhanh, tự động từ GitHub
- ✅ Hỗ trợ Python tốt
- ✅ Có region Singapore (gần VN)

**Cách deploy:**
```bash
# Chạy script tự động
./deploy-railway.sh

# Hoặc thủ công:
1. Truy cập https://railway.app
2. Login với GitHub
3. New Project > Deploy from GitHub repo
4. Chọn repo: TuanChienVu/banking-test-generator
5. Railway tự động detect và deploy
```

### Option 2: Render (Miễn phí 750 giờ/tháng)
**Ưu điểm:**
- ✅ 750 giờ miễn phí/tháng
- ✅ Không cần credit card
- ✅ Auto-deploy từ GitHub

**Cách deploy:**
```bash
1. Truy cập https://render.com
2. New > Web Service
3. Connect GitHub repo
4. Chọn repo banking-test-generator
5. Settings:
   - Build Command: pip install -r requirements.txt
   - Start Command: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
6. Create Web Service
```

### Option 3: Local + Ngrok (Test nhanh)
**Ưu điểm:**
- ✅ Chạy ngay lập tức
- ✅ Không cần đăng ký
- ✅ Debug dễ dàng

**Cách chạy:**
```bash
# Chạy script tự động
./deploy-local.sh

# Hoặc thủ công:
# Terminal 1 - Backend
cd backend
uvicorn app:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
python -m http.server 3000

# Terminal 3 - Ngrok
ngrok http 8000
```

### Option 4: Vercel (Frontend) + Railway (Backend)
**Ưu điểm:**
- ✅ Frontend load nhanh với Vercel CDN
- ✅ Backend stable với Railway
- ✅ Custom domain dễ dàng

**Cách deploy:**

**Backend (Railway):**
```bash
./deploy-railway.sh
# Copy URL backend (ví dụ: https://banking-api.railway.app)
```

**Frontend (Vercel):**
```bash
1. Cài Vercel CLI: npm i -g vercel
2. cd frontend
3. vercel
4. Follow prompts
5. Set environment variable:
   API_URL = [Railway backend URL]
```

## 🌐 Custom Domain Setup (vutuanchien.dev)

### Với GitHub Pages:
```bash
1. Deploy frontend lên GitHub Pages
2. Tạo file CNAME: echo "vutuanchien.dev" > frontend/CNAME
3. Commit và push

4. Tại Name.com:
   - Add A records:
     185.199.108.153
     185.199.109.153
     185.199.110.153
     185.199.111.153
   - Add CNAME: www -> TuanChienVu.github.io
```

### Với Vercel:
```bash
1. Trong Vercel Dashboard > Domains
2. Add domain: vutuanchien.dev
3. Follow DNS instructions từ Vercel
```

### Với Railway:
```bash
1. Railway Dashboard > Settings > Domains
2. Add custom domain: api.vutuanchien.dev
3. Add CNAME tại Name.com:
   api -> [railway-domain].railway.app
```

## 📦 File Structure
```
clean_project/
├── backend/           # FastAPI backend
│   ├── app.py       # Main API
│   └── utils.py     # Helper functions
├── frontend/         # Static frontend
│   ├── index.html   # Main page
│   ├── app.js       # Frontend logic
│   ├── styles.css   # Styling
│   └── config.js    # API configuration
├── model_trained/    # AI model files
├── requirements.txt  # Python dependencies
├── runtime.txt      # Python version
├── Procfile         # Heroku config (optional)
├── railway.json     # Railway config
├── render.yaml      # Render config
└── vercel.json      # Vercel config
```

## 🔧 Environment Variables

### Backend cần:
```
PYTHON_VERSION=3.11
MODEL_PATH=model_trained
PORT=8000 (Railway/Render tự set)
```

### Frontend cần:
```javascript
// frontend/config.js
const API_URL = 'https://your-backend-url.com';
```

## 🚨 Troubleshooting

### Lỗi "Module not found":
```bash
pip install -r requirements.txt
```

### Lỗi CORS:
```python
# Đã config sẵn trong backend/app.py
# Kiểm tra API_URL trong frontend/config.js
```

### Model file quá lớn:
```bash
# Sử dụng Git LFS
git lfs track "*.pt" "*.pth" "*.bin"
git add .gitattributes
git commit -m "Track model files with LFS"
```

### Port đã sử dụng:
```bash
# Kill process đang dùng port
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

## 📱 Testing

### Local:
```bash
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Production:
```bash
# Check health: curl https://your-api.railway.app/health
# Test generate: Use frontend or Postman
```

## 💡 Tips

1. **Railway** nhanh nhất cho MVP/demo
2. **Render** ổn định cho production
3. **Vercel + Railway** tốt nhất cho performance
4. Luôn test local trước khi deploy
5. Monitor logs để debug: `railway logs` hoặc Render dashboard

## 📞 Support

- GitHub Issues: https://github.com/TuanChienVu/banking-test-generator/issues
- Email: tuanchienvu@gmail.com
- Domain: vutuanchien.dev

---
Made with ❤️ by Tuan Chien Vu
