FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY fast-api-server /app/fast-api-server


EXPOSE 8000

CMD ["python", "fast-api-server/main.py"]
