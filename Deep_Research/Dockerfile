# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies if any (none needed for basic fastAPI but listing just in case)
# RUN apt-get update && apt-get install -y --no-install-recommends ...

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
COPY pyproject.toml .
# We might need to copy README/LICENSE if references exist in pyproject.toml
COPY README.md . 

# Install dependencies and the package itself
# Install requirements first for caching
RUN pip install --no-cache-dir -r requirements.txt
# Copy the source code
COPY src/ ./src/
# Install the package in non-editable mode
RUN pip install .

# Copy the app entrypoint and service
COPY app.py .
COPY service.py .

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
