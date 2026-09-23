FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# The Python package contains bindings, while libeccodes0 provides a native
# fallback on Linux.  ca-certificates is required for the NOAA HTTPS archive.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.lock \
    && python -m eccodes selfcheck

COPY . .
RUN python -m pip install --no-deps --editable . \
    && useradd --create-home --uid 10001 twinturbo \
    && chown -R twinturbo:twinturbo /app

USER twinturbo

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=4)" || exit 1

CMD ["python", "scripts/run.py"]
