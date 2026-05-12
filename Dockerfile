# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install necessary build tools if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api/ api/
COPY model/ model/
COPY monitoring/ monitoring/
COPY training/ configs/ training/configs/

# Expose port 8080 for SageMaker
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application on port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
