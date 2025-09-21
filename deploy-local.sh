#!/bin/bash

# Script để chạy local với ngrok
echo "🚀 Starting Banking Test Generator locally..."

# Kiểm tra ngrok đã cài chưa
if ! command -v ngrok &> /dev/null; then
    echo "📦 Installing ngrok..."
    brew install ngrok
fi

# Chạy backend
echo "🔧 Starting backend server..."
python -m uvicorn backend.app:app --reload --port 8000 &
BACKEND_PID=$!

# Chạy frontend 
echo "🌐 Starting frontend server..."
python -m http.server 3001 --directory frontend &
FRONTEND_PID=$!

# Expose qua ngrok
echo "🌍 Exposing via ngrok..."
ngrok http 8000 --log-level=info &
NGROK_PID=$!

echo "✅ Services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Ngrok PID: $NGROK_PID"

echo ""
echo "📝 Access your app at:"
echo "- Frontend: http://localhost:3001"
echo "- Backend: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo ""
echo "Note: Ngrok requires free account registration at https://ngrok.com"
echo "After registration, run: ngrok config add-authtoken YOUR_TOKEN"

# Trap để dọn dẹp khi tắt
trap "kill $BACKEND_PID $FRONTEND_PID $NGROK_PID 2>/dev/null" EXIT

# Giữ script chạy
wait
