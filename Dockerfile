FROM python:3.11-slim

# Install system dependencies (needed for OpenCV caching & system bindings)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements and install
COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy frontend and backend directories
COPY backend /code/backend
COPY frontend /code/frontend

# Create cache directories
RUN mkdir -p /code/local_temp

# Set Hugging Face port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "backend"]
