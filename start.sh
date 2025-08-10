#!/bin/bash

# Start script for the Unified Trading Bot Current Market Dashboard

echo "🚀 Starting Unified Trading Bot Current Market Dashboard"
echo ""

# Check if Python and Node.js are available
echo "Checking dependencies..."
python3 --version || { echo "Python 3 is required"; exit 1; }
node --version || { echo "Node.js is required"; exit 1; }
npm --version || { echo "npm is required"; exit 1; }

echo "✅ Dependencies found"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd "$(dirname "$0")"
pip install -r requirements.txt

echo ""

# Install Node.js dependencies (if not already installed)
echo "📦 Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi

echo ""

# Start backend server in background
echo "🔧 Starting backend server..."
cd ../backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend server started successfully"
else
    echo "❌ Failed to start backend server"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎯 Current Market Dashboard is ready!"
echo ""
echo "📊 Backend API: http://localhost:8000"
echo "🔗 API Documentation: http://localhost:8000/docs"
echo "📈 Frontend: Coming soon (React app setup)"
echo ""
echo "💡 To view market data, try:"
echo "   curl http://localhost:8000/market-data"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running and handle shutdown
trap 'echo ""; echo "🛑 Shutting down services..."; kill $BACKEND_PID 2>/dev/null; echo "✅ Services stopped"; exit 0' INT

# Wait for backend process
wait $BACKEND_PID