from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_MODULE_PATHS = [
    os.path.join(CURRENT_DIR, "..", "build", "Release"),
    os.path.join(CURRENT_DIR, "..", "build"),
]

for module_path in POSSIBLE_MODULE_PATHS:
    if os.path.exists(module_path):
        sys.path.insert(0, module_path)
        if os.name == "nt":
            try:
                os.add_dll_directory(module_path)
            except (AttributeError, FileNotFoundError):
                pass

if os.name == "nt":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    if os.path.exists(cuda_bin):
        try:
            os.add_dll_directory(cuda_bin)
        except (AttributeError, FileNotFoundError):
            pass

import cuda_lstm_ops


app = FastAPI(
    title="CUDA Sequence Inference Engine",
    description=(
        "Single-layer LSTM and last-query self-attention demonstration"
    ),
)
Instrumentator().instrument(app).expose(app)


DEMO_INPUT_SIZE = 4
DEMO_HIDDEN_SIZE = 8
DEMO_SEQ_LEN = 5
DEMO_SEED = 7

BENCH_INPUT_SIZE = 1024
BENCH_HIDDEN_SIZE = 1024
BENCH_SEQ_LEN = 50
BENCH_SEED = 2025
BENCH_GPU_ITERATIONS = 5


def make_lstm_data(
    input_size: int,
    hidden_size: int,
    seed: int,
    weight_scale: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    input_weights = rng.uniform(
        -weight_scale,
        weight_scale,
        size=(input_size, 4 * hidden_size),
    ).astype(np.float32)
    recurrent_weights = rng.uniform(
        -weight_scale,
        weight_scale,
        size=(hidden_size, 4 * hidden_size),
    ).astype(np.float32)
    biases = rng.uniform(
        -0.4 * weight_scale,
        0.4 * weight_scale,
        size=(4 * hidden_size,),
    ).astype(np.float32)
    return input_weights, recurrent_weights, biases


print("Initializing the interactive CUDA engine...")
demo_engine = cuda_lstm_ops.Inference(
    DEMO_INPUT_SIZE,
    DEMO_HIDDEN_SIZE,
    DEMO_SEQ_LEN,
)

demo_input_weights, demo_recurrent_weights, demo_biases = make_lstm_data(
    DEMO_INPUT_SIZE,
    DEMO_HIDDEN_SIZE,
    DEMO_SEED,
    weight_scale=0.35,
)
demo_engine.load_lstm_weights(
    demo_input_weights,
    demo_recurrent_weights,
    demo_biases,
)

demo_rng = np.random.default_rng(DEMO_SEED + 1)
demo_Wq = demo_rng.uniform(
    -1.5,
    1.5,
    size=(DEMO_INPUT_SIZE, DEMO_HIDDEN_SIZE),
).astype(np.float32)
demo_Wk = demo_rng.uniform(
    -1.5,
    1.5,
    size=(DEMO_INPUT_SIZE, DEMO_HIDDEN_SIZE),
).astype(np.float32)
demo_Wv = demo_rng.uniform(
    -0.25,
    0.25,
    size=(DEMO_INPUT_SIZE, DEMO_HIDDEN_SIZE),
).astype(np.float32)
demo_engine.load_attention_weights(demo_Wq, demo_Wk, demo_Wv)

# Warm up one-time CUDA library initialization without leaking recurrent state.
demo_warmup_input = np.zeros(
    (DEMO_SEQ_LEN, DEMO_INPUT_SIZE),
    dtype=np.float32,
)
demo_engine.run_inference(demo_warmup_input)
demo_engine.run_self_attention(demo_warmup_input)


class InferenceRequest(BaseModel):
    input_data: list[float]


@dataclass
class BenchmarkState:
    gpu: object
    cpu: object
    input_data: np.ndarray
    warmed_up: bool = False


benchmark_state: BenchmarkState | None = None
demo_execution_lock = threading.Lock()
benchmark_initialization_lock = threading.Lock()
benchmark_execution_lock = threading.Lock()


def get_benchmark_state() -> BenchmarkState:
    global benchmark_state

    if benchmark_state is not None:
        return benchmark_state

    with benchmark_initialization_lock:
        if benchmark_state is not None:
            return benchmark_state

        print("Initializing benchmark engines and deterministic weights...")
        input_weights, recurrent_weights, biases = make_lstm_data(
            BENCH_INPUT_SIZE,
            BENCH_HIDDEN_SIZE,
            BENCH_SEED,
        )
        rng = np.random.default_rng(BENCH_SEED + 1)
        input_data = rng.uniform(
            -0.5,
            0.5,
            size=(BENCH_SEQ_LEN, BENCH_INPUT_SIZE),
        ).astype(np.float32)

        gpu = cuda_lstm_ops.Inference(
            BENCH_INPUT_SIZE,
            BENCH_HIDDEN_SIZE,
            BENCH_SEQ_LEN,
        )
        cpu = cuda_lstm_ops.CpuInference(
            BENCH_INPUT_SIZE,
            BENCH_HIDDEN_SIZE,
            BENCH_SEQ_LEN,
        )
        gpu.load_lstm_weights(input_weights, recurrent_weights, biases)
        cpu.load_lstm_weights(input_weights, recurrent_weights, biases)

        benchmark_state = BenchmarkState(
            gpu=gpu,
            cpu=cpu,
            input_data=input_data,
        )
        return benchmark_state


def validate_input(values: list[float]) -> np.ndarray:
    expected = DEMO_SEQ_LEN * DEMO_INPUT_SIZE
    if len(values) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {expected} input values",
        )
    return np.asarray(values, dtype=np.float32).reshape(
        DEMO_SEQ_LEN,
        DEMO_INPUT_SIZE,
    )


def timed_gpu_call(method, input_data: np.ndarray):
    started = time.perf_counter()
    output, compute_ms = method(input_data)
    end_to_end_ms = (time.perf_counter() - started) * 1_000.0
    return np.asarray(output), float(compute_ms), end_to_end_ms


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "CUDA sequence inference engine ready",
        "device": demo_engine.device_name(),
    }


@app.get("/model")
def model_info():
    return {
        "device": demo_engine.device_name(),
        "lstm": {
            "input_size": DEMO_INPUT_SIZE,
            "hidden_size": DEMO_HIDDEN_SIZE,
            "sequence_length": DEMO_SEQ_LEN,
            "type": "single-layer LSTM",
        },
        "attention": {
            "type": "single-head, last-query self-attention",
            "returned_attention_weights": DEMO_SEQ_LEN,
        },
    }


@app.post("/predict/lstm")
def predict_lstm(request: InferenceRequest):
    input_data = validate_input(request.input_data)
    with demo_execution_lock:
        output, execution_time_ms = demo_engine.run_inference(input_data)
    return {
        "output": np.asarray(output).tolist(),
        "execution_time_ms": float(execution_time_ms),
    }


@app.post("/predict/attention")
def predict_attention(request: InferenceRequest):
    input_data = validate_input(request.input_data)
    with demo_execution_lock:
        output, weights, execution_time_ms = demo_engine.run_self_attention(
            input_data
        )
    return {
        "output": np.asarray(output).tolist(),
        "attention_weights": np.asarray(weights).tolist(),
        "execution_time_ms": float(execution_time_ms),
    }


@app.post("/benchmark")
def benchmark():
    with benchmark_execution_lock:
        state = get_benchmark_state()

        if not state.warmed_up:
            for _ in range(2):
                state.gpu.run_inference(state.input_data)
                state.gpu.run_inference_fp16(state.input_data)
            state.warmed_up = True

        cpu_output, cpu_time_ms = state.cpu.run_inference(state.input_data)
        cpu_output = np.asarray(cpu_output)

        fp32_results = [
            timed_gpu_call(state.gpu.run_inference, state.input_data)
            for _ in range(BENCH_GPU_ITERATIONS)
        ]
        fp16_results = [
            timed_gpu_call(state.gpu.run_inference_fp16, state.input_data)
            for _ in range(BENCH_GPU_ITERATIONS)
        ]

        fp32_output = fp32_results[-1][0]
        fp16_output = fp16_results[-1][0]
        fp32_error = float(np.max(np.abs(fp32_output - cpu_output)))
        fp16_error = float(np.max(np.abs(fp16_output - cpu_output)))
        verified = fp32_error <= 1e-4 and fp16_error <= 2e-2

        if not verified:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Benchmark output validation failed",
                    "fp32_max_abs_error": fp32_error,
                    "fp16_max_abs_error": fp16_error,
                },
            )

        gpu_time_ms = statistics.median(
            result[1] for result in fp32_results
        )
        gpu_time_fp16_ms = statistics.median(
            result[1] for result in fp16_results
        )
        gpu_end_to_end_ms = statistics.median(
            result[2] for result in fp32_results
        )
        gpu_end_to_end_fp16_ms = statistics.median(
            result[2] for result in fp16_results
        )

        return {
            "gpu_time_ms": gpu_time_ms,
            "gpu_time_fp16_ms": gpu_time_fp16_ms,
            "gpu_end_to_end_ms": gpu_end_to_end_ms,
            "gpu_end_to_end_fp16_ms": gpu_end_to_end_fp16_ms,
            "cpu_time_ms": float(cpu_time_ms),
            "speedup": float(cpu_time_ms) / gpu_time_ms,
            "speedup_fp16": float(cpu_time_ms) / gpu_time_fp16_ms,
            "fp32_max_abs_error": fp32_error,
            "fp16_max_abs_error": fp16_error,
            "verified": verified,
            "iterations": BENCH_GPU_ITERATIONS,
            "device": state.gpu.device_name(),
            "configuration": {
                "input_size": BENCH_INPUT_SIZE,
                "hidden_size": BENCH_HIDDEN_SIZE,
                "sequence_length": BENCH_SEQ_LEN,
                "parameter_count": (
                    4
                    * BENCH_HIDDEN_SIZE
                    * (BENCH_INPUT_SIZE + BENCH_HIDDEN_SIZE)
                    + 4 * BENCH_HIDDEN_SIZE
                ),
            },
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
