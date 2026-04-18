@echo off
echo Starting GuardRail backend...
start "GuardRail Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting GuardRail frontend...
cd frontend
start "GuardRail Frontend" cmd /k "npm run dev"
cd ..

echo.
echo GuardRail is running:
echo   Backend API:  http://localhost:8000
echo   Frontend:     http://localhost:5173
echo   API Docs:     http://localhost:8000/docs
echo.
