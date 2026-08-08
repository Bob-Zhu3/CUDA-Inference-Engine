"""Reproducible CPU vs CUDA benchmark for the single-layer LSTM engine."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_MODULE_PATHS = [
    os.path.join(CURRENT_DIR, "build", "Release"),
    os.path.join(CURRENT_DIR, "build"),
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

try:
    import cuda_lstm_ops
except ImportError as error:
    raise SystemExit(
        "Could not import cuda_lstm_ops. Build the project first.\n"
        f"Original error: {error}"
    ) from error


@dataclass
class RunResult:
    output: np.ndarray
    compute_ms: float
    end_to_end_ms: float


def make_data(
    input_size: int,
    hidden_size: int,
    seq_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic, numerically stable synthetic model data."""

    rng = np.random.default_rng(seed)
    weight_scale = 0.025

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
        -0.01,
        0.01,
        size=(4 * hidden_size,),
    ).astype(np.float32)
    input_data = rng.uniform(
        -0.5,
        0.5,
        size=(seq_len, input_size),
    ).astype(np.float32)

    return input_weights, recurrent_weights, biases, input_data


def timed_call(method, input_data: np.ndarray) -> RunResult:
    started = time.perf_counter()
    output, compute_ms = method(input_data)
    end_to_end_ms = (time.perf_counter() - started) * 1_000.0
    return RunResult(
        output=np.asarray(output),
        compute_ms=float(compute_ms),
        end_to_end_ms=end_to_end_ms,
    )


def run_benchmark(args: argparse.Namespace) -> int:
    print(
        "Configuration: "
        f"input={args.input_size}, hidden={args.hidden_size}, "
        f"steps={args.seq_len}, seed={args.seed}"
    )

    input_weights, recurrent_weights, biases, input_data = make_data(
        args.input_size,
        args.hidden_size,
        args.seq_len,
        args.seed,
    )

    gpu = cuda_lstm_ops.Inference(
        args.input_size,
        args.hidden_size,
        args.seq_len,
    )
    cpu = cuda_lstm_ops.CpuInference(
        args.input_size,
        args.hidden_size,
        args.seq_len,
    )

    print(f"GPU: {gpu.device_name()}")
    print("Loading model weights once...")
    gpu.load_lstm_weights(input_weights, recurrent_weights, biases)
    cpu.load_lstm_weights(input_weights, recurrent_weights, biases)

    # The engines now own their copies, so the large source arrays can be
    # released before inference.
    del input_weights, recurrent_weights, biases

    print(f"Warming up each GPU path {args.warmups} time(s)...")
    for _ in range(args.warmups):
        gpu.run_inference(input_data)
        gpu.run_inference_fp16(input_data)

    print("Running the single-threaded CPU reference...")
    cpu_result = timed_call(cpu.run_inference, input_data)

    fp32_results = [
        timed_call(gpu.run_inference, input_data)
        for _ in range(args.gpu_iterations)
    ]
    fp16_results = [
        timed_call(gpu.run_inference_fp16, input_data)
        for _ in range(args.gpu_iterations)
    ]

    fp32_error = float(
        np.max(np.abs(fp32_results[-1].output - cpu_result.output))
    )
    fp16_error = float(
        np.max(np.abs(fp16_results[-1].output - cpu_result.output))
    )

    fp32_compute_ms = statistics.median(
        result.compute_ms for result in fp32_results
    )
    fp16_compute_ms = statistics.median(
        result.compute_ms for result in fp16_results
    )
    fp32_end_to_end_ms = statistics.median(
        result.end_to_end_ms for result in fp32_results
    )
    fp16_end_to_end_ms = statistics.median(
        result.end_to_end_ms for result in fp16_results
    )

    fp32_ok = fp32_error <= args.fp32_tolerance
    fp16_ok = fp16_error <= args.fp16_tolerance

    print("\nCorrectness check")
    print(
        f"  FP32 max absolute difference: {fp32_error:.8f} "
        f"({'PASS' if fp32_ok else 'FAIL'})"
    )
    print(
        f"  FP16 max absolute difference: {fp16_error:.8f} "
        f"({'PASS' if fp16_ok else 'FAIL'})"
    )

    print("\nMedian timing")
    print(f"  CPU reference compute: {cpu_result.compute_ms:.2f} ms")
    print(
        f"  CUDA FP32 compute:      {fp32_compute_ms:.2f} ms "
        f"({cpu_result.compute_ms / fp32_compute_ms:.1f}x)"
    )
    print(
        f"  CUDA FP16 compute:      {fp16_compute_ms:.2f} ms "
        f"({cpu_result.compute_ms / fp16_compute_ms:.1f}x)"
    )
    print(f"  CUDA FP32 end-to-end:   {fp32_end_to_end_ms:.2f} ms")
    print(f"  CUDA FP16 end-to-end:   {fp16_end_to_end_ms:.2f} ms")

    if not fp32_ok or not fp16_ok:
        print("\nBenchmark rejected because output validation failed.")
        return 1

    print("\nBenchmark passed: timings correspond to numerically valid output.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--gpu-iterations", type=int, default=5)
    parser.add_argument("--fp32-tolerance", type=float, default=1e-4)
    parser.add_argument("--fp16-tolerance", type=float, default=2e-2)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_benchmark(parse_args()))
