#!/bin/bash
# Start FastAPI backend on port 7861 (internal)
uvicorn hf_server:app --host 0.0.0.0 --port 7861 &

# Wait for backend to be ready
sleep 5

# Start Gradio UI on port 7860 (HF public port)
python ui.py
