FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraudservice ./fraudservice
COPY models ./models
ENV MODEL_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "fraudservice.api:app", "--host", "0.0.0.0", "--port", "8000"]
