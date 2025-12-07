# Face Authentication Application Startup Script (Windows PowerShell)
# This script sets up and runs both the FastAPI backend and React frontend

Write-Host "Starting Face Authentication Application..." -ForegroundColor Cyan
Write-Host ""

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ============================================
# BACKEND SETUP
# ============================================
Write-Host "Setting up Python backend..." -ForegroundColor Yellow

# Navigate to API directory
Set-Location api

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# Install requirements
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install MCP server
Write-Host "Installing MCP server..."
pip install mcp
pip install -e "$ScriptDir\api\assistant-mcp-server"

# Start MCP server in background
Write-Host "Starting MCP server..." -ForegroundColor Green
$McpProcess = Start-Process -FilePath ".\venv\Scripts\assistant-mcp-server.exe" `
    -NoNewWindow -PassThru
Write-Host "MCP server started (PID: $($McpProcess.Id))" -ForegroundColor Green

# Start FastAPI backend in background
Write-Host "Starting FastAPI backend server..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000" `
    -NoNewWindow -PassThru

Write-Host "Backend server started (PID: $($BackendProcess.Id))" -ForegroundColor Green
Write-Host ""

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# ============================================
# FRONTEND SETUP
# ============================================
Write-Host "Setting up React frontend..." -ForegroundColor Yellow

# Navigate to frontend directory
Set-Location "$ScriptDir\ui\app"

# Install npm dependencies
Write-Host "Installing npm dependencies..."
npm install

# Start frontend dev server
Write-Host "Starting frontend development server..." -ForegroundColor Green
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Application will open in your browser..." -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Open browser
Start-Process "http://localhost:5173"

# Start dev server
npm run dev

# ============================================
# CLEANUP
# ============================================
Write-Host ""
Write-Host "Shutting down..." -ForegroundColor Yellow

# Stop the backend and MCP processes
Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $McpProcess.Id -Force -ErrorAction SilentlyContinue

Write-Host "Application stopped" -ForegroundColor Green

