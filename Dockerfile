# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY text_classifier ./text_classifier
# CPU-only torch keeps the image ~1.5 GB smaller.
RUN pip install --no-cache-dir --prefix=/install \
      --extra-index-url https://download.pytorch.org/whl/cpu ".[models,api]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN useradd --create-home --uid 10001 app && mkdir /models && chown app /models
WORKDIR /app
COPY --from=builder /install /usr/local
USER app
VOLUME ["/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "text_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
