FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-10000} \
    --server.address=0.0.0.0 \
    --server.headless=true