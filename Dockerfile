FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloud.txt ./requirements-cloud.txt
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .
RUN python scripts/download_fonts.py
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0"]
