FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY EmotionDetection ./EmotionDetection
COPY templates ./templates
COPY server.py .
RUN useradd -m app && chown -R app /app
USER app
ENV PORT=8080
EXPOSE 8080
CMD exec gunicorn -b 0.0.0.0:${PORT} -w 2 --timeout 30 server:app
