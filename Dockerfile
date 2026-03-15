FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for pygame and X11 forwarding
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxcursor1 \
    libxrandr2 \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command
CMD ["python", "train.py"]
