FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3

# the feeder corrupts each batch with the same injector as the demo scripts
COPY scripts/inject_errors.py /app/inject_errors.py
COPY feeder/feeder.py /app/feeder.py

CMD ["python", "-u", "/app/feeder.py"]
