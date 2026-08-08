FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cuda-nsight-systems-12-8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip3 install --no-cache-dir -r backend/requirements.txt

COPY . .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel

EXPOSE 8000

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
