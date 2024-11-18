# Stage 1: Build environment with Poetry and dependencies
FROM python:3.10.15-slim as builder

LABEL maintainer="Soutrik soutrik1991@gmail.com" \
      description="Docker image for running a Python app with dependencies managed by Poetry."

# Install Poetry and necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Poetry to the PATH explicitly
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory to /app
WORKDIR /app

# Copy pyproject.toml and poetry.lock to install dependencies
COPY pyproject.toml poetry.lock /app/

# Configure Poetry environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install dependencies without installing the package itself
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Stage 2: Runtime environment
FROM python:3.10.15-slim as runner

# Install curl for health check script
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application source code and necessary files
COPY src /app/src
COPY configs /app/configs
COPY .project-root /app/.project-root
COPY main.py /app/main.py

# Copy virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the client files
COPY run_client.sh /app/run_client.sh

# Set the working directory to /app
WORKDIR /app

# Set the environment path to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "-m", "main"]
