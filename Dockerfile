FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY wsgi.py .

# 必须单 worker（调度器是进程内的内存状态），用多线程承接并发请求
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "64", "--timeout", "300", "wsgi:app"]
