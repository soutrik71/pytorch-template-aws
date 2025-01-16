# Use Python base image
FROM python:3.12

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/app

# Set working directory
WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY model-server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the checkpoints folder
COPY model-server/server.py ./server.py
COPY src ./src
COPY checkpoints ./checkpoints
COPY configs ./configs
COPY .project-root ./.project-root

# Expose the app's port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
