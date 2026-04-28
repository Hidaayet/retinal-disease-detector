FROM python:3.10-slim

# install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy requirements first for caching
COPY requirements.txt .

# install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy app files
COPY app/ ./app/
COPY data/best_model.pth ./data/best_model.pth

# expose port
EXPOSE 7860

# run with gunicorn instead of Flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app.app:app"]