# Base image for Python app
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
# Default to running the UI; override command in Render if using separate services.
CMD ["gunicorn", "ui.app:app", "--bind", "0.0.0.0:10000"]
