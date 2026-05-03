# Base image — slim means minimal, no unnecessary packages
# This is the "OS" inside our container
FROM python:3.12-slim

# Set working directory inside container
# All commands run from here
WORKDIR /app

# Give uv more time for large wheel downloads in container builds.
ENV UV_HTTP_TIMEOUT=120

# Install uv for fast package installation
RUN pip install uv

# Copy dependency files first
# Docker caches each line — if requirements don't change,
# this layer is cached and packages aren't reinstalled every build
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of the code
# We do this AFTER installing packages so code changes
# don't invalidate the package cache
COPY . .

# Create data directory for PDFs and ChromaDB
RUN mkdir -p data

# Expose port 8000 so outside world can reach FastAPI
EXPOSE 8000

# Command to run when container starts
CMD ["uv", "run", "python", "main.py"]