# Stage 1: Base image with CUDA 12.2, cuDNN 9, and minimal runtime for PyTorch
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as base

LABEL maintainer="Soutrik soutrik1991@gmail.com" \
      description="Base Docker image for running a Python app with Poetry and GPU support."

# Install necessary system dependencies, including Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    curl \
    git \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python --version

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Configure Poetry environment
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory to /app
WORKDIR /app

# Copy pyproject.toml and poetry.lock to install dependencies
COPY pyproject.toml poetry.lock /app/

# Install Python dependencies without building the app itself
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install --only main --no-root

# Stage 2: Build stage for the application
FROM base as builder

# Copy application source code and necessary files
COPY src /app/src
COPY configs /app/configs
COPY .project-root /app/.project-root
COPY main.py /app/main.py

# Stage 3: Final runtime stage
FROM base as runner

# Copy application source code and dependencies from the builder stage
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/.project-root /app/.project-root
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/.venv /app/.venv

# Copy client files
COPY run_client.sh /app/run_client.sh

# Set the working directory to /app
WORKDIR /app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Install PyTorch with CUDA 12.2 support (adjusted for compatibility)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Default command to run the application
CMD ["python", "-m", "main"]
