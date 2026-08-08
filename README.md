# CUDA Sequence Inference Engine

A compact C++/CUDA demonstration of two sequence-model operations:

- A real single-layer LSTM with input and recurrent projections.
- Single-head, last-query self-attention.

cuBLAS performs the large matrix operations. Custom CUDA kernels apply the
LSTM gates, calculate attention scores, normalize them with a stable softmax,
and build the attended summary. The native engine is exposed to Python through
pybind11 and served through a FastAPI/React dashboard.

The project uses deterministic synthetic inputs and weights. It demonstrates
the inference math and CPU/GPU performance; it is not a trained prediction
model.

## What the LSTM computes

For each time step:

```text
gates = input_weights × current_input
      + recurrent_weights × previous_hidden_state
      + biases

cell_t   = forget_gate × cell_(t-1)
         + input_gate × candidate_content

hidden_t = output_gate × tanh(cell_t)
```

The four gate groups are stored in this order:

```text
forget, input/write, candidate, output/reveal
```

Every inference begins from zero cell and hidden state. Model weights are
loaded onto the GPU once and reused.

## What attention computes

The attention path projects every input step into query, key, and value
vectors. It compares the final step's query with every key, turns those scores
into percentages that sum to one, and returns:

- One attention percentage per input step.
- One weighted summary vector.

This intentionally computes the output for the final query only. It is not a
full Transformer layer or a complete all-to-all attention matrix.

## Dashboard

The browser demo:

1. Generates five input vectors containing four numbers each.
2. Displays the LSTM's final eight-number hidden state.
3. Displays the five real attention percentages and the attended summary.
4. Runs a validated CPU/FP32/FP16 benchmark.

The benchmark uses a `1024 → 1024` LSTM over 50 steps, or 8,392,704 model
parameters. Before reporting speed, it compares both GPU outputs with the
single-threaded CPU reference and rejects the run if the numerical difference
is too large.

An example native Windows run on an NVIDIA GeForce RTX 3060 Laptop GPU
produced:

```text
CPU reference:  approximately 496 ms
CUDA FP32:       approximately 7.7 ms  (64x)
CUDA FP16:       approximately 6.0 ms  (83x)
FP32 max error:  below 0.000001
FP16 max error:  below 0.0001
```

Exact results depend on the GPU, driver, clock state, and host CPU. The speedup
is versus the project's straightforward single-threaded CPU reference, not an
optimized library such as oneDNN or cuDNN.

## Architecture

```text
React page (localhost:8080)
        |
        | /api
        v
Nginx reverse proxy
        |
        v
FastAPI server (localhost:8000)
        |
        v
pybind11 native module
        |
        v
C++ / CUDA / cuBLAS

Prometheus scrapes FastAPI request metrics every 15 seconds.
Grafana starts with Prometheus connected and a request dashboard provisioned.
```

## Requirements

- NVIDIA GPU
- Compatible NVIDIA driver
- Docker Desktop or Docker Engine
- NVIDIA Container Toolkit / Docker GPU support

## Run the complete stack

```bash
cd CudaInferenceEngine
docker compose up --build
```

Then open:

- App: <http://localhost:8080>
- API documentation: <http://localhost:8000/docs>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (`admin` / `admin`)

Grafana monitors API traffic and response time. It does not claim to collect
GPU temperature or utilization.

## Run the benchmark directly

Inside the backend container:

```bash
docker compose exec inference-engine python3 benchmark_script.py
```

Or after a local native build:

```bash
cd CudaInferenceEngine
python benchmark_script.py
```

The script uses fixed random seeds, loads weights once, performs GPU warm-ups,
validates outputs, and reports median GPU compute and end-to-end times.

## Profile with NVIDIA Nsight Systems

In the running Docker stack:

```bash
docker compose exec inference-engine bash profile.sh
```

Or directly on native Windows:

```powershell
cd CudaInferenceEngine
.\profile.ps1
```

This creates:

```text
lstm_benchmark_report_validated.nsys-rep
```

The profiling build includes CUDA source line information. The report should
show cuBLAS operations followed by legal multi-block LSTM update kernels; CUDA
and cuBLAS failures now raise errors instead of silently producing timings.

## Build the native module locally

```bash
cd CudaInferenceEngine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

CMake downloads pybind11 `v2.11.1` during initial configuration.

## Main files

```text
src/inference.cu             CUDA LSTM, FP16, and attention implementation
src/cpu_reference.cpp        single-threaded correctness/performance reference
src/bindings.cpp             NumPy/pybind11 interface
backend/server.py            FastAPI routes and validated web benchmark
frontend/src/App.tsx         inference demonstration
frontend/src/BenchmarkChart.tsx
benchmark_script.py          reproducible command-line benchmark
profile.sh                   Nsight Systems profiling command
profile.ps1                  native Windows Nsight profiling command
docker-compose.yml           app, API, Prometheus, and Grafana
```

## Intentional limitations

- One LSTM layer
- One sequence per inference call
- Synthetic rather than trained weights
- Only the final LSTM hidden state is returned
- Attention returns only the final query's result
- CPU reference is intentionally straightforward and single-threaded
- FP16 stores states and weights in half precision while accumulating matrix
  products and evaluating gates in regular precision

These boundaries keep the engine small enough to understand and explain while
still demonstrating recurrent state, custom GPU kernels, cuBLAS integration,
mixed precision, native Python bindings, profiling, and a containerized web
stack.
