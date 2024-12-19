FROM public.ecr.aws/docker/library/python:3.12-slim

# Copy the Lambda adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV PYPI_MIRROR=https://pypi.org/simple

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Pre-download large packages (e.g., triton) and cache them
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 triton==3.0.0

# Install the remaining requirements
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=120 --retries=5

# Copy only the necessary files
COPY main.py ./main.py
COPY src ./src
COPY .env ./.env
COPY checkpoints ./checkpoints
COPY configs ./configs
COPY .project-root ./.project-root

# Expose the port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
