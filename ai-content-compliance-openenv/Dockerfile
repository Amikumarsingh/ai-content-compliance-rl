# Dockerfile for Hugging Face Spaces
# Based on the production Dockerfile with HF-specific configuration

FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create results directory
RUN mkdir -p results && chmod 755 results

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start both FastAPI (on 7861) and Gradio UI (on 7860)
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
