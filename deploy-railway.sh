#!/bin/bash

echo "🚀 Deploying to Railway..."

# Kiểm tra Railway CLI
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    brew install railway
fi

# Login Railway
echo "🔑 Please login to Railway..."
railway login

# Tạo project mới
echo "📦 Creating new Railway project..."
railway init

# Link với GitHub repo
echo "🔗 Linking with GitHub repository..."
railway link

# Set environment variables
echo "⚙️ Setting environment variables..."
railway vars set PYTHON_VERSION=3.11
railway vars set MODEL_PATH=model_trained
railway vars set PORT=8000

# Deploy
echo "🚂 Deploying to Railway..."
railway up

# Lấy URL của app
echo "🌐 Getting deployment URL..."
railway open

echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Copy the Railway backend URL"
echo "2. Update frontend/config.js with the new URL"
echo "3. Deploy frontend to GitHub Pages or Vercel"
echo ""
echo "💡 To view logs: railway logs"
echo "💡 To redeploy: railway up"
