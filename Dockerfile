FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install -U pip && pip install -r requirements.txt

COPY . .

# Default to linear futures DB path
ENV CATEGORY=linear \
    DB_PATH=data/bybit_linear.sqlite3

VOLUME ["/app/data", "/app/logs"]

CMD ["python", "main.py"]


