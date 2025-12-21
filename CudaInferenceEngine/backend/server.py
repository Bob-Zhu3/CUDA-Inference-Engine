import sys
import os
import random
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, "..", "build", "Release"),
    os.path.join(current_dir, "..", "build")
]

for p in possible_paths:
    if os.path.exists(p):
        sys.path.append(p)
        if os.name == 'nt':
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

if os.name == 'nt':
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    if os.path.exists(cuda_path):
        try:
            os.add_dll_directory(cuda_path)
        except Exception:
            pass

import cuda_lstm_ops

app = FastAPI()
Instrumentator().instrument(app).expose(app)

input_size = 4
hidden_size = 8
seq_len = 5

print("initializing cuda inference engine...")
engine = cuda_lstm_ops.Inference(input_size, hidden_size, seq_len)

bench_input_size = 1024
bench_hidden_size = 4096
bench_seq_len = 50

print("initializing benchmark engines...")
bench_engine = cuda_lstm_ops.Inference(bench_input_size, bench_hidden_size, bench_seq_len)
bench_cpu_engine = cuda_lstm_ops.CpuInference(bench_input_size, bench_hidden_size, bench_seq_len)

def random_list(size):
    return [random.random() - 0.5 for _ in range(size)]

print("generating random model weights...")
weights = random_list(4 * hidden_size * input_size)
biases = random_list(4 * hidden_size)

print("generating benchmark weights...")
bench_weights = random_list(4 * bench_hidden_size * bench_input_size)
bench_biases = random_list(4 * bench_hidden_size)

wq = random_list(input_size * hidden_size)
wk = random_list(input_size * hidden_size)
wv = random_list(input_size * hidden_size)

class InferenceRequest(BaseModel):
    input_data: list[float]

@app.get("/")
def root():
    return {"status": "online", "message": "cuda inference engine ready"}

@app.post("/predict/lstm")
def predict_lstm(req: InferenceRequest):
    print("received lstm inference request")
    expected_len = seq_len * input_size
    if len(req.input_data) != expected_len:
        print(f"error: expected {expected_len} floats, got {len(req.input_data)}")
        return {"error": f"expected {expected_len} floats"}
    
    output, _ = engine.run_inference(req.input_data, weights, biases)
    print("lstm inference complete")
    return {"output": output}

@app.post("/predict/attention")
def predict_attention(req: InferenceRequest):
    print("received attention inference request")
    expected_len = seq_len * input_size
    if len(req.input_data) != expected_len:
        print(f"error: expected {expected_len} floats, got {len(req.input_data)}")
        return {"error": f"expected {expected_len} floats"}
    
    output = engine.run_self_attention(req.input_data, wq, wk, wv)
    print("attention inference complete")
    return {"output": output}

@app.post("/benchmark")
def benchmark():
    print("running benchmark...")
    input_data = random_list(bench_seq_len * bench_input_size)
    
    _, gpu_time = bench_engine.run_inference(input_data, bench_weights, bench_biases)
    _, gpu_time_fp16 = bench_engine.run_inference_fp16(input_data, bench_weights, bench_biases)
    _, cpu_time = bench_cpu_engine.run_inference(input_data, bench_weights, bench_biases)
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
    speedup_fp16 = cpu_time / gpu_time_fp16 if gpu_time_fp16 > 0 else 0.0
    
    return {
        "gpu_time_ms": gpu_time,
        "gpu_time_fp16_ms": gpu_time_fp16,
        "cpu_time_ms": cpu_time,
        "speedup": speedup,
        "speedup_fp16": speedup_fp16
    }

if __name__ == "__main__":
    print("starting uvicorn server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)