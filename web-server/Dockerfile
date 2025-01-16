# Use Python base image
FROM python:3.12

# Set environment variables
ENV PORT=9000
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app

# Set working directory
WORKDIR /app

# Copy requirements.txt to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and necessary files
COPY web-server/server.py ./server.py

# Expose the port the app will run on
EXPOSE 9000

# Default command to run the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
