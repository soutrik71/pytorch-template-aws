# Use an official Python base image
FROM python:3.10.15-slim

# Set environment variables for Poetry
ENV POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1

# Install Poetry
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Poetry files for dependency installation
COPY pyproject.toml poetry.lock ./

# Install dependencies with Poetry (without development dependencies)
RUN poetry install --no-root --only main

# Copy the application source code
COPY src/ src/
COPY app/ app/
COPY main.py  .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application with uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
