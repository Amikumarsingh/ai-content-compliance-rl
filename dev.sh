#!/bin/bash
# Start GuardRail full stack in development mode

echo "Starting GuardRail backend..."
cd "$(dirname "$0")"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting GuardRail frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo ""
echo "GuardRail is running:"
echo "  Backend API:  http://localhost:8000"
echo "  Frontend:     http://localhost:5173"
echo "  API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
