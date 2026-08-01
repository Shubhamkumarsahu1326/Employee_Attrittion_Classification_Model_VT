# ==========================================
# STAGE 1: Dependency Builder Layer
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ==========================================
# STAGE 2: Production Runtime Execution Layer
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy entire structured folders to preserve relative imports natively
COPY model/ ./model/
COPY config/ ./config/
COPY src/ ./src/
COPY app/ ./app/

# Set Python Path variable to include the app working directory
ENV PYTHONPATH=/app

# EXPOSE FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000
EXPOSE 8501

# FIXED: Target the application explicitly through the folder module tree pathing (folder.subfolder.file:variable)
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
