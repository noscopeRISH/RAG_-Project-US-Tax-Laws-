# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid downloading large CUDA-enabled version (~3GB vs ~300MB)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements.txt and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Minimal runner image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime libraries (FAISS requires libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local

# Configure PATH to include user packages and set env vars
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Copy application source files
COPY src/ /app/src/
COPY api_main.py /app/

# Bake pre-built FAISS index, BM25 cache, and documents into the image
COPY data/ /app/data/

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860

# Start server using api_main.py
CMD ["python", "api_main.py"]
