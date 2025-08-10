#!/bin/bash

# Start script for the Unified Trading Bot Modal Dashboard

echo "🚀 Starting Unified Trading Bot Modal Dashboard"
echo ""

# Check if Python is available
echo "Checking dependencies..."
python3 --version || { echo "Python 3 is required"; exit 1; }

echo "✅ Dependencies found"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd "$(dirname "$0")"
pip install -r requirements.txt

echo ""

# Check if Modal is available for cloud deployment
if command -v modal &> /dev/null; then
    echo "🔧 Modal CLI found. You can deploy to Modal cloud with:"
    echo "   modal deploy modal_app.py"
    echo ""
fi

# Start local dashboard
echo "🔧 Starting local dashboard..."
python modal_app.py &
DASHBOARD_PID=$!

# Wait for dashboard to start
sleep 5

# Check if dashboard is running
if curl -s http://localhost:8050/ > /dev/null; then
    echo "✅ Dashboard started successfully"
else
    echo "❌ Failed to start dashboard"
    kill $DASHBOARD_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎯 AAPL Stock Dashboard is ready!"
echo ""
echo "📊 Local Dashboard: http://localhost:8050"
echo "📈 Live AAPL Data: Updates every 5 minutes"
echo ""
echo "💡 For cloud deployment:"
echo "   1. Setup Modal: python -m modal setup"
echo "   2. Deploy: modal deploy modal_app.py"
echo ""
echo "Press Ctrl+C to stop the dashboard"

# Keep script running and handle shutdown
trap 'echo ""; echo "🛑 Shutting down dashboard..."; kill $DASHBOARD_PID 2>/dev/null; echo "✅ Dashboard stopped"; exit 0' INT

# Wait for dashboard process
wait $DASHBOARD_PID