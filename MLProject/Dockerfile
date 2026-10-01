FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY . .

# Expose port for model serving
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1

# Command to run MLflow model serving
CMD ["mlflow", "models", "serve", "-m", "runs:/latest/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]
