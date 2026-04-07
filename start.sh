#!/bin/bash
# Start FastAPI backend on port 8000 (internal)
uvicorn hf_server:app --host 0.0.0.0 --port 8000 &

# Wait for backend to be ready
sleep 3

# Start Gradio UI on port 7860 (public)
python ui.py
