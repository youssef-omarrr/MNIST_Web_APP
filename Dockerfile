# 1. Base image
FROM python:3.10

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project files
COPY . .

# 5. Expose the Flask default port
EXPOSE 5000

# 6. Run the app
CMD ["python", "app.py"]

# 7. Build image and run container
# docker build -t flask-app .
# docker run -p 5000:5000 -v "D:\Codess & Projects\MNIST_Web_APP:/app" flask-app
