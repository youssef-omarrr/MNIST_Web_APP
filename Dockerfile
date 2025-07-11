# syntax=docker/dockerfile:1

# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 4. Expose port
EXPOSE 5000

# 5. Run the app
CMD ["python", "app.py"]