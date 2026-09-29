#!/bin/bash

# ============================================================
#  Smart Attendance System - Start All Services
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║       Smart Attendance System Launcher       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Backend API (FastAPI on port 8000) ────────────────────
echo "▶ Starting Backend API on http://localhost:8000 ..."
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR/backend/backend-api' && ./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000\""

sleep 2

# ── 2. ML Service (FastAPI on port 8001) ─────────────────────
echo "▶ Starting ML Service   on http://localhost:8001 ..."
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR/backend/ml-service' && ./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8001\""

sleep 2

# ── 3. Frontend (Vite on port 5173) ──────────────────────────
echo "▶ Starting Frontend     on http://localhost:5173 ..."
osascript -e "tell app \"Terminal\" to do script \"cd '$PROJECT_DIR/frontend' && npm run dev\""

sleep 5

# ── Open in browser ───────────────────────────────────────────
echo ""
echo "✅ All services started! Opening app in browser..."
open http://localhost:5173

echo ""
echo "┌─────────────────────────────────────────────────┐"
echo "│  Frontend   →  http://localhost:5173            │"
echo "│  Backend    →  http://localhost:8000/docs       │"
echo "│  ML Service →  http://localhost:8001/docs       │"
echo "└─────────────────────────────────────────────────┘"
echo ""
echo "Close the Terminal tabs to stop the services."
