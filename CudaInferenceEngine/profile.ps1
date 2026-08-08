$ErrorActionPreference = "Stop"

$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
$cmake = if ($cmakeCommand) {
    $cmakeCommand.Source
} else {
    @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\CMake\bin\cmake.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $cmake) {
    throw "CMake was not found on PATH or in the standard Visual Studio locations."
}

& $cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
& $cmake --build build --config Release --parallel

$nsysCandidates = @(
    "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe",
    "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.1\target-windows-x64\nsys.exe"
)
$nsys = $nsysCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $nsys) {
    $nsysCommand = Get-Command nsys -ErrorAction SilentlyContinue
    if (-not $nsysCommand) {
        throw "NVIDIA Nsight Systems (nsys.exe) was not found."
    }
    $nsys = $nsysCommand.Source
}

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
$python = if ($pythonLauncher) {
    & $pythonLauncher.Source -3.10 -c "import sys; print(sys.executable)"
} else {
    & python -c "import sys; print(sys.executable)"
}
$python = $python.Trim()
if (-not $python -or -not (Test-Path $python)) {
    throw "The real Python interpreter path could not be resolved."
}

& $nsys profile `
    --trace=cuda,cublas `
    --sample=none `
    --cpuctxsw=none `
    --stats=true `
    --force-overwrite=true `
    --output=lstm_benchmark_report_validated `
    $python benchmark_script.py

if ($LASTEXITCODE -ne 0) {
    throw "Nsight Systems profiling failed with exit code $LASTEXITCODE."
}
