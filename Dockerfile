FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY app ./app
COPY scripts ./scripts
COPY reports ./reports
COPY eval ./eval

# The index is built from the manuals when the container is first started with an empty /app/data volume.
EXPOSE 8000
CMD ["sh", "-c", "[ -f data/index/default/faiss.index ] || (python scripts/fetch_manuals.py && python scripts/build_index.py); uvicorn rag.service.api:app --host 0.0.0.0 --port 8000"]
