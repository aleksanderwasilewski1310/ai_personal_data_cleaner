# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7
# --- Stage 1: Extract the official uv binary ---
# We start from the official lightweight Python image
FROM python:3.11-slim AS final

# Optimize Python execution behavior inside the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install curl to fetch the uv installer, then install uv,
# and clean up apt lists immediately to keep the layer minimal.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

# Add the installed uv binary to the system PATH so it can be called directly
ENV PATH="/root/.local/bin/:$PATH"

# Leverage Docker cache layers effectively:
# Copy only the compiled requirements file first.
COPY requirements.txt ./

# Install project dependencies into the global system environment using uv pip
# --system: Installs packages directly into the global Python environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# FastAPI application defaults to port 8000
EXPOSE 8000

# Create a system group and a non-root user without a home directory or login shell
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch the execution context to the secure non-root user
USER appuser

# Command to launch the Uvicorn ASGI server
CMD ["uvicorn", "main:APP", "--host", "0.0.0.0", "--port", "8000"]
