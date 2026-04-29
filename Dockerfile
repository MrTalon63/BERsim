FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY master_server.py .
COPY plot_results.py .
COPY smart_scheduler.py .

EXPOSE 8000

CMD ["python3", "master_server.py"]