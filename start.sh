#!/bin/bash
# Start FastAPI server on port 7860 (public — OpenEnv validator pings this port)
exec uvicorn hf_server:app --host 0.0.0.0 --port 7860
