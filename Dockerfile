FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV VIRTUAL_ENV=/opt/viavsr-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    python3 \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade pip

RUN python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url "${PYTORCH_INDEX_URL}"

COPY . .

RUN python -m pip install -e ".[dev]"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/viavsr/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
