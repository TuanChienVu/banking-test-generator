#!/bin/bash

echo "🚀 Starting Banking Test Generator locally..."

# Kiểm tra port 8000 có đang bị sử dụng không
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port 8000 is already in use. Killing process..."
    lsof -ti:8000 | xargs kill -9
fi

# Kiểm tra port 3001 có đang bị sử dụng không
if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port 3001 is already in use. Killing process..."
    lsof -ti:3001 | xargs kill -9
fi

# Chạy backend
echo "🔧 Starting backend server on port 8000..."
python -m uvicorn server:app --reload --port 8000 &
BACKEND_PID=$!

# Đợi backend khởi động
sleep 3

# Chạy frontend 
echo "🌐 Starting frontend server on port 3001..."
python -m http.server 3001 --directory web &
FRONTEND_PID=$!

# Đợi frontend khởi động
sleep 2

echo ""
echo "✅ Services started successfully!"
echo ""
echo "📝 Access your app at:"
echo "┌─────────────────────────────────────────────────┐"
echo "│ 🌐 Frontend: http://localhost:3001              │"
echo "│ 🔧 Backend:  http://localhost:8000              │"
echo "│ 📚 API Docs: http://localhost:8000/docs         │"
echo "└─────────────────────────────────────────────────┘"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Mở browser
if command -v open &> /dev/null; then
    echo "Opening browser..."
    sleep 2
    open http://localhost:3001
fi

# Trap để dọn dẹp khi tắt
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Services stopped.'" EXIT

# Giữ script chạy
wait
