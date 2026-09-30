# Build from the challenge/ directory: docker build -t chess-model-challenge .
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
COPY frontend/public ./public
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHALLENGE_STATIC_DIR=/app/static \
    PORT=8000
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 challenge
COPY backend/api.py ./api.py
COPY --from=frontend /build/dist ./static
USER challenge
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"
# FORWARDED_ALLOW_IPS: set to your proxy's address (or * when only the platform proxy can reach the container)
# so rate limits apply per visitor instead of per proxy.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port \"${PORT}\" --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\" --no-server-header --timeout-graceful-shutdown 20"]
