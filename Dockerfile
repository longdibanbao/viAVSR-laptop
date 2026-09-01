FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python3.11 -m pip install --upgrade pip

RUN python3.11 -m pip install -e ".[dev]"

EXPOSE 8501

CMD ["python3.11", "-m", "streamlit", "run", "src/viavsr/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
