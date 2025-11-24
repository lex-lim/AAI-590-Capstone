# Face Authentication Application

This directory contains the complete Face Authentication application with both backend API and frontend web interface.

## Quick Start

### Mac / Linux
```bash
./start_linux.sh
```

### Windows
```powershell
.\start_windows.ps1
```

## What the Scripts Do

Both startup scripts will automatically:
1. ✅ Create and activate a Python virtual environment
2. ✅ Install all Python dependencies from `requirements.txt`
3. ✅ Start the FastAPI backend server on `http://localhost:8000`
4. ✅ Install all npm dependencies
5. ✅ Start the React frontend development server
6. ✅ Open your default web browser to `http://localhost:5173`

## Manual Setup

If you prefer to run components separately:

### Backend (FastAPI)
```bash
cd api
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (React + Vite)
```bash
cd ui/app
npm install
npm run dev
```

## Application URLs

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Stopping the Application

Press `Ctrl+C` in the terminal to stop both servers. The scripts will automatically clean up background processes.

## Requirements

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Webcam** for face authentication features

## Project Structure

```
app/
├── api/              # FastAPI backend
│   ├── main.py       # API endpoints
│   ├── face_classifier.py
│   ├── models.py
│   └── requirements.txt
├── ui/app/           # React frontend
│   ├── src/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
├── start.sh          # Mac/Linux startup script
└── start.ps1         # Windows startup script
```

