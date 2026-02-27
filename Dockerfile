FROM python:3.12-slim

# Install ghostscript + qpdf, then verify they work
RUN apt-get update && \
    apt-get install -y --no-install-recommends ghostscript qpdf && \
    rm -rf /var/lib/apt/lists/* && \
    gs --version && \
    qpdf --version

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Longer timeout (300s) for large PDFs that need GS processing
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 300
