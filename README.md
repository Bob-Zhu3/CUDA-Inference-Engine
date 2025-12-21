# CUDA Inference Engine

An LSTM and Self-Attention inference engine using CUDA and C++. This project demonstrates the massive speedup of GPU acceleration compared to CPU implementation for deep learning.

## Features

*   C++ implementation of LSTM cells and Self-Attention mechanisms using cuBLAS.
*   Benchmarking of CPU and GPU inference speeds.

<img width="1424" height="1037" alt="Description" src="https://github.com/user-attachments/assets/f3601d8f-715a-4d0b-ab53-f698c445be12" />
<img width="1841" height="751" alt="runinference" src="https://github.com/user-attachments/assets/7e9de253-9b36-4031-905f-f9cf681e2070" />
<img width="1466" height="507" alt="performancebenchmarkpng" src="https://github.com/user-attachments/assets/8a9a4dc5-f9cc-4271-85e2-1e8623566c17" />
<img width="1435" height="617" alt="CPUvsGPU2" src="https://github.com/user-attachments/assets/f5d0e3aa-acb7-47ec-8c72-5a2b97c39112" />


## Prerequisites

*   **NVIDIA GPU**
*   **NVIDIA Drivers** installed on the host machine.
*   **Docker Desktop** (Windows) or Docker Engine (Linux).
*   **NVIDIA Container Toolkit** (to allow Docker to access the GPU).

## Quickstart

1.  **Clone**
    ```bash
    git clone https://github.com/Bob-Zhu3/CUDA-Inference-Engine.git
    cd CUDA-Inference-Engine
    ```

2.  **Run with Docker Compose**
    ```bash
    cd CudaInferenceEngine
    docker-compose up -d --build
    ```

3.  **App**
    *   Frontend: [http://localhost:8080](http://localhost:8080)
    *   Backend: [http://localhost:8000](http://localhost:8000)
    *   Monitoring: Prometheus [http://localhost:9090](http://localhost:9090) Grafana: [http://localhost:3000](http://localhost:3000) (Login: admin/admin)

## Model Architecture

A standard LSTM model with:
*   **Input Size:** 1024
*   **Hidden Size:** 4096
*   **Sequence Length:** 50
*   **Total Parameters:** ~84 million
