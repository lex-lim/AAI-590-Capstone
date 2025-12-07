#!/bin/bash

# Face Authentication Application Startup Script (Mac/Linux)
# This script sets up and runs both the FastAPI backend and React frontend

set -e  # Exit on error

echo "Starting Face Authentication Application..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================
# BACKEND SETUP
# ============================================
echo "Setting up Python backend..."

# Navigate to API directory
cd api

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install MCP server
echo "Installing MCP server..."
pip install mcp
pip install -e "$SCRIPT_DIR/api/assistant-mcp-server"

# Start MCP server in background
echo "Starting MCP server..."
python -m assistant_mcp.server &
MCP_PID=$!
echo "MCP server started (PID: $MCP_PID)"

# Start FastAPI backend in background
echo "Starting FastAPI backend server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Backend server started (PID: $BACKEND_PID)"
echo ""

# Wait a moment for backend to start
sleep 3

# ============================================
# FRONTEND SETUP
# ============================================
echo "Setting up React frontend..."

# Navigate to frontend directory
cd "$SCRIPT_DIR/ui/app"

# Install npm dependencies
echo "Installing npm dependencies..."
npm install

# Start frontend dev server (will open browser automatically)
echo "Starting frontend development server..."
echo ""
echo "=========================================="
echo "Application will open in your browser..."
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "=========================================="
echo ""

# Open browser and start dev server
npm run dev -- --open

# ============================================
# CLEANUP
# ============================================
# When frontend is stopped (Ctrl+C), also stop the backend and MCP server
echo ""
echo "Shutting down..."
kill $BACKEND_PID 2>/dev/null || true
kill $MCP_PID 2>/dev/null || true
echo "Application stopped"

