FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio>=4.0.0

COPY . .

RUN mkdir -p results && chmod 755 results

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV EVALUATOR_PROVIDER=mock
# Gradio calls the FastAPI backend on port 8000 internally
ENV API_BASE_URL=http://localhost:8000

# Start script runs both FastAPI (port 8000) and Gradio (port 7860)
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
