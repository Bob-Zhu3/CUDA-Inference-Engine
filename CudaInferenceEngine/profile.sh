#!/usr/bin/env bash
set -euo pipefail

echo "Building an optimized profiling target..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

echo "Running the validated benchmark under Nsight Systems..."
nsys profile \
    --trace=cuda,cublas,osrt \
    --sample=none \
    --cpuctxsw=none \
    --stats=true \
    --force-overwrite=true \
    --output=lstm_benchmark_report_validated \
    python3 benchmark_script.py --warmups 2 --gpu-iterations 5

echo "Profiling complete."
echo "Report: lstm_benchmark_report_validated.nsys-rep"
