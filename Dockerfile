FROM python:3.12-slim

# Don't write .pyc files, don't buffer stdout so logs show immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY ridecourt/ ridecourt/
COPY web/ web/
COPY data/ data/
COPY scripts/ scripts/

# Create a non-root user and switch to it.
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
RUN chown -R app:app /app
USER app

# Cloud Run sets PORT=8080 by default; honour it.
ENV PORT=8080
EXPOSE 8080

# uvicorn with a single worker (in-memory state). Cloud Run handles scaling.
CMD ["sh", "-c", "uvicorn ridecourt.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
