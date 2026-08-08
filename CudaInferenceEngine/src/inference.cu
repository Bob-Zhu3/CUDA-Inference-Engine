#include "inference.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

void check_cuda(cudaError_t status,
                const char* expression,
                const char* file,
                int line)
{
    if (status == cudaSuccess)
    {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line
            << " (" << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

const char* cublas_status_name(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
        return "CUBLAS_STATUS_UNKNOWN";
    }
}

void check_cublas(cublasStatus_t status,
                  const char* expression,
                  const char* file,
                  int line)
{
    if (status == CUBLAS_STATUS_SUCCESS)
    {
        return;
    }

    std::ostringstream message;
    message << "cuBLAS call failed: " << expression << " at " << file << ':'
            << line << " (" << cublas_status_name(status) << ')';
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

#define CUBLAS_CHECK(expression) \
    check_cublas((expression), #expression, __FILE__, __LINE__)

class CudaEventTimer
{
public:
    CudaEventTimer() : start_event(nullptr), stop_event(nullptr)
    {
        CUDA_CHECK(cudaEventCreate(&start_event));
        try
        {
            CUDA_CHECK(cudaEventCreate(&stop_event));
        }
        catch (...)
        {
            cudaEventDestroy(start_event);
            start_event = nullptr;
            throw;
        }
    }

    ~CudaEventTimer()
    {
        if (start_event != nullptr)
        {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr)
        {
            cudaEventDestroy(stop_event);
        }
    }

    void start()
    {
        CUDA_CHECK(cudaEventRecord(start_event));
    }

    float stop()
    {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
};

__device__ float sigmoid_device(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void lstm_update_kernel(const float* z,
                                   float* cell_state,
                                   float* hidden_state,
                                   int hidden_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size)
    {
        return;
    }

    const float forget_gate = sigmoid_device(z[index]);
    const float input_gate = sigmoid_device(z[index + hidden_size]);
    const float candidate = tanhf(z[index + 2 * hidden_size]);
    const float output_gate = sigmoid_device(z[index + 3 * hidden_size]);

    const float new_cell =
        forget_gate * cell_state[index] + input_gate * candidate;
    cell_state[index] = new_cell;
    hidden_state[index] = output_gate * tanhf(new_cell);
}

__global__ void lstm_update_fp16_kernel(const __half* z,
                                        __half* cell_state,
                                        __half* hidden_state,
                                        int hidden_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size)
    {
        return;
    }

    const float forget_gate = sigmoid_device(__half2float(z[index]));
    const float input_gate =
        sigmoid_device(__half2float(z[index + hidden_size]));
    const float candidate =
        tanhf(__half2float(z[index + 2 * hidden_size]));
    const float output_gate =
        sigmoid_device(__half2float(z[index + 3 * hidden_size]));

    const float previous_cell = __half2float(cell_state[index]);
    const float new_cell =
        forget_gate * previous_cell + input_gate * candidate;
    const float new_hidden = output_gate * tanhf(new_cell);

    cell_state[index] = __float2half(new_cell);
    hidden_state[index] = __float2half(new_hidden);
}

__global__ void float_to_half_kernel(const float* source,
                                     __half* destination,
                                     size_t count)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        destination[index] = __float2half(source[index]);
    }
}

__global__ void half_to_float_kernel(const __half* source,
                                     float* destination,
                                     size_t count)
{
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        destination[index] = __half2float(source[index]);
    }
}

__global__ void attention_scores_kernel(const float* queries,
                                        const float* keys,
                                        float* scores,
                                        int time_steps,
                                        int model_size,
                                        int query_index)
{
    const int key_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_index >= time_steps)
    {
        return;
    }

    const float* query = queries + query_index * model_size;
    const float* key = keys + key_index * model_size;

    float dot_product = 0.0f;
    for (int coordinate = 0; coordinate < model_size; ++coordinate)
    {
        dot_product += query[coordinate] * key[coordinate];
    }

    scores[key_index] =
        dot_product / sqrtf(static_cast<float>(model_size));
}

// The demo intentionally limits attention to at most 1024 time steps. One
// block is then enough for a clear, stable softmax implementation.
__global__ void attention_softmax_kernel(float* scores, int count)
{
    extern __shared__ float scratch[];
    const int index = threadIdx.x;

    scratch[index] =
        index < count ? scores[index] : -3.402823466e+38F;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (index < stride)
        {
            scratch[index] = fmaxf(scratch[index], scratch[index + stride]);
        }
        __syncthreads();
    }

    const float maximum = scratch[0];
    const float exponential =
        index < count ? expf(scores[index] - maximum) : 0.0f;
    scratch[index] = exponential;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (index < stride)
        {
            scratch[index] += scratch[index + stride];
        }
        __syncthreads();
    }

    const float sum = scratch[0];
    if (index < count)
    {
        scores[index] = exponential / sum;
    }
}

__global__ void attention_weighted_sum_kernel(const float* values,
                                              const float* attention_weights,
                                              float* output,
                                              int time_steps,
                                              int model_size)
{
    const int coordinate = blockIdx.x * blockDim.x + threadIdx.x;
    if (coordinate >= model_size)
    {
        return;
    }

    float sum = 0.0f;
    for (int time = 0; time < time_steps; ++time)
    {
        sum += attention_weights[time] *
               values[static_cast<size_t>(time) * model_size + coordinate];
    }
    output[coordinate] = sum;
}

int blocks_for(size_t count, int threads)
{
    return static_cast<int>((count + threads - 1) / threads);
}

int next_power_of_two(int value)
{
    int result = 1;
    while (result < value)
    {
        result *= 2;
    }
    return result;
}

} // namespace

Inference::Inference(int in_dim, int hid_dim, int seq_len)
    : input_size(in_dim),
      hidden_size(hid_dim),
      time_steps(seq_len),
      d_model(hid_dim),
      lstm_weights_loaded(false),
      attention_weights_loaded(false),
      d_input(nullptr),
      d_input_weights(nullptr),
      d_recurrent_weights(nullptr),
      d_biases(nullptr),
      d_hidden_state(nullptr),
      d_cell_state(nullptr),
      d_z(nullptr),
      d_Wq(nullptr),
      d_Wk(nullptr),
      d_Wv(nullptr),
      d_Q(nullptr),
      d_K(nullptr),
      d_V(nullptr),
      d_attention_scores(nullptr),
      d_attention_output(nullptr),
      d_input_fp16(nullptr),
      d_input_weights_fp16(nullptr),
      d_recurrent_weights_fp16(nullptr),
      d_biases_fp16(nullptr),
      d_hidden_state_fp16(nullptr),
      d_cell_state_fp16(nullptr),
      d_z_fp16(nullptr),
      cublas_handle(nullptr)
{
    if (input_size <= 0 || hidden_size <= 0 || time_steps <= 0)
    {
        throw std::invalid_argument(
            "input_dim, hidden_dim, and seq_len must all be positive");
    }

    try
    {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        allocate_memory();
    }
    catch (...)
    {
        release_memory();
        throw;
    }
}

Inference::~Inference()
{
    release_memory();
}

void Inference::allocate_memory()
{
    const size_t input_count =
        static_cast<size_t>(time_steps) * input_size;
    const size_t input_weight_count =
        static_cast<size_t>(4) * hidden_size * input_size;
    const size_t recurrent_weight_count =
        static_cast<size_t>(4) * hidden_size * hidden_size;
    const size_t gate_count = static_cast<size_t>(4) * hidden_size;
    const size_t attention_weight_count =
        static_cast<size_t>(input_size) * d_model;
    const size_t attention_projection_count =
        static_cast<size_t>(time_steps) * d_model;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input),
                          input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_weights),
                          input_weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_recurrent_weights),
                          recurrent_weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_biases),
                          gate_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hidden_state),
                          hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cell_state),
                          hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_z),
                          gate_count * sizeof(float)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Wq),
                          attention_weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Wk),
                          attention_weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Wv),
                          attention_weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Q),
                          attention_projection_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_K),
                          attention_projection_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_V),
                          attention_projection_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_attention_scores),
                          time_steps * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_attention_output),
                          d_model * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_input_fp16,
                          input_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_input_weights_fp16,
                          input_weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_recurrent_weights_fp16,
                          recurrent_weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_biases_fp16,
                          gate_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_hidden_state_fp16,
                          hidden_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_cell_state_fp16,
                          hidden_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_z_fp16,
                          gate_count * sizeof(__half)));
}

void Inference::release_memory() noexcept
{
    cudaFree(d_input);
    cudaFree(d_input_weights);
    cudaFree(d_recurrent_weights);
    cudaFree(d_biases);
    cudaFree(d_hidden_state);
    cudaFree(d_cell_state);
    cudaFree(d_z);

    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_attention_scores);
    cudaFree(d_attention_output);

    cudaFree(d_input_fp16);
    cudaFree(d_input_weights_fp16);
    cudaFree(d_recurrent_weights_fp16);
    cudaFree(d_biases_fp16);
    cudaFree(d_hidden_state_fp16);
    cudaFree(d_cell_state_fp16);
    cudaFree(d_z_fp16);

    if (cublas_handle != nullptr)
    {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

void Inference::load_lstm_weights(
    const std::vector<float>& host_input_weights,
    const std::vector<float>& host_recurrent_weights,
    const std::vector<float>& host_biases)
{
    const size_t gate_count = static_cast<size_t>(4) * hidden_size;
    const size_t input_weight_count = gate_count * input_size;
    const size_t recurrent_weight_count = gate_count * hidden_size;

    if (host_input_weights.size() != input_weight_count)
    {
        throw std::invalid_argument(
            "input_weights must contain input_size * 4 * hidden_size values");
    }
    if (host_recurrent_weights.size() != recurrent_weight_count)
    {
        throw std::invalid_argument(
            "recurrent_weights must contain hidden_size * 4 * hidden_size values");
    }
    if (host_biases.size() != gate_count)
    {
        throw std::invalid_argument(
            "biases must contain 4 * hidden_size values");
    }

    CUDA_CHECK(cudaMemcpy(d_input_weights,
                          host_input_weights.data(),
                          input_weight_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_recurrent_weights,
                          host_recurrent_weights.data(),
                          recurrent_weight_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases,
                          host_biases.data(),
                          gate_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    float_to_half_kernel<<<blocks_for(input_weight_count, threads), threads>>>(
        d_input_weights,
        static_cast<__half*>(d_input_weights_fp16),
        input_weight_count);
    CUDA_CHECK(cudaGetLastError());

    float_to_half_kernel<<<blocks_for(recurrent_weight_count, threads), threads>>>(
        d_recurrent_weights,
        static_cast<__half*>(d_recurrent_weights_fp16),
        recurrent_weight_count);
    CUDA_CHECK(cudaGetLastError());

    float_to_half_kernel<<<blocks_for(gate_count, threads), threads>>>(
        d_biases,
        static_cast<__half*>(d_biases_fp16),
        gate_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    lstm_weights_loaded = true;
}

void Inference::load_attention_weights(
    const std::vector<float>& host_Wq,
    const std::vector<float>& host_Wk,
    const std::vector<float>& host_Wv)
{
    if (time_steps > 1024)
    {
        throw std::invalid_argument(
            "the simplified attention path supports at most 1024 time steps");
    }

    const size_t weight_count =
        static_cast<size_t>(input_size) * d_model;
    if (host_Wq.size() != weight_count ||
        host_Wk.size() != weight_count ||
        host_Wv.size() != weight_count)
    {
        throw std::invalid_argument(
            "each attention weight array must contain input_size * hidden_size values");
    }

    CUDA_CHECK(cudaMemcpy(d_Wq,
                          host_Wq.data(),
                          weight_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk,
                          host_Wk.data(),
                          weight_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv,
                          host_Wv.data(),
                          weight_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    attention_weights_loaded = true;
}

void Inference::validate_input(const std::vector<float>& input) const
{
    const size_t expected =
        static_cast<size_t>(time_steps) * input_size;
    if (input.size() != expected)
    {
        throw std::invalid_argument(
            "input must contain seq_len * input_size values");
    }
}

void Inference::run_inference(const std::vector<float>& host_input,
                              std::vector<float>& host_output,
                              float* execution_time_ms)
{
    if (!lstm_weights_loaded)
    {
        throw std::runtime_error(
            "LSTM weights must be loaded before inference");
    }
    validate_input(host_input);

    const size_t input_count =
        static_cast<size_t>(time_steps) * input_size;
    CUDA_CHECK(cudaMemcpy(d_input,
                          host_input.data(),
                          input_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hidden_state, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cell_state, 0, hidden_size * sizeof(float)));

    CudaEventTimer timer;
    timer.start();

    constexpr int threads = 256;
    const int state_blocks = blocks_for(hidden_size, threads);
    const float alpha = 1.0f;
    const float beta = 1.0f;

    for (int time = 0; time < time_steps; ++time)
    {
        const float* input_at_time =
            d_input + static_cast<size_t>(time) * input_size;

        CUDA_CHECK(cudaMemcpyAsync(d_z,
                                   d_biases,
                                   static_cast<size_t>(4) * hidden_size *
                                       sizeof(float),
                                   cudaMemcpyDeviceToDevice));

        CUBLAS_CHECK(cublasSgemv(cublas_handle,
                                 CUBLAS_OP_N,
                                 4 * hidden_size,
                                 input_size,
                                 &alpha,
                                 d_input_weights,
                                 4 * hidden_size,
                                 input_at_time,
                                 1,
                                 &beta,
                                 d_z,
                                 1));

        CUBLAS_CHECK(cublasSgemv(cublas_handle,
                                 CUBLAS_OP_N,
                                 4 * hidden_size,
                                 hidden_size,
                                 &alpha,
                                 d_recurrent_weights,
                                 4 * hidden_size,
                                 d_hidden_state,
                                 1,
                                 &beta,
                                 d_z,
                                 1));

        lstm_update_kernel<<<state_blocks, threads>>>(
            d_z, d_cell_state, d_hidden_state, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    const float elapsed_ms = timer.stop();
    if (execution_time_ms != nullptr)
    {
        *execution_time_ms = elapsed_ms;
    }

    host_output.resize(hidden_size);
    CUDA_CHECK(cudaMemcpy(host_output.data(),
                          d_hidden_state,
                          hidden_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void Inference::run_inference_fp16(const std::vector<float>& host_input,
                                   std::vector<float>& host_output,
                                   float* execution_time_ms)
{
    if (!lstm_weights_loaded)
    {
        throw std::runtime_error(
            "LSTM weights must be loaded before inference");
    }
    validate_input(host_input);

    const size_t input_count =
        static_cast<size_t>(time_steps) * input_size;
    CUDA_CHECK(cudaMemcpy(d_input,
                          host_input.data(),
                          input_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    float_to_half_kernel<<<blocks_for(input_count, threads), threads>>>(
        d_input,
        static_cast<__half*>(d_input_fp16),
        input_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(
        d_hidden_state_fp16, 0, hidden_size * sizeof(__half)));
    CUDA_CHECK(cudaMemset(
        d_cell_state_fp16, 0, hidden_size * sizeof(__half)));

    CudaEventTimer timer;
    timer.start();

    const int state_blocks = blocks_for(hidden_size, threads);
    const float alpha = 1.0f;
    const float beta = 1.0f;

    for (int time = 0; time < time_steps; ++time)
    {
        const __half* input_at_time =
            static_cast<const __half*>(d_input_fp16) +
            static_cast<size_t>(time) * input_size;

        CUDA_CHECK(cudaMemcpyAsync(d_z_fp16,
                                   d_biases_fp16,
                                   static_cast<size_t>(4) * hidden_size *
                                       sizeof(__half),
                                   cudaMemcpyDeviceToDevice));

        CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  4 * hidden_size,
                                  1,
                                  input_size,
                                  &alpha,
                                  d_input_weights_fp16,
                                  CUDA_R_16F,
                                  4 * hidden_size,
                                  input_at_time,
                                  CUDA_R_16F,
                                  input_size,
                                  &beta,
                                  d_z_fp16,
                                  CUDA_R_16F,
                                  4 * hidden_size,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  4 * hidden_size,
                                  1,
                                  hidden_size,
                                  &alpha,
                                  d_recurrent_weights_fp16,
                                  CUDA_R_16F,
                                  4 * hidden_size,
                                  d_hidden_state_fp16,
                                  CUDA_R_16F,
                                  hidden_size,
                                  &beta,
                                  d_z_fp16,
                                  CUDA_R_16F,
                                  4 * hidden_size,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        lstm_update_fp16_kernel<<<state_blocks, threads>>>(
            static_cast<const __half*>(d_z_fp16),
            static_cast<__half*>(d_cell_state_fp16),
            static_cast<__half*>(d_hidden_state_fp16),
            hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    const float elapsed_ms = timer.stop();
    if (execution_time_ms != nullptr)
    {
        *execution_time_ms = elapsed_ms;
    }

    half_to_float_kernel<<<state_blocks, threads>>>(
        static_cast<const __half*>(d_hidden_state_fp16),
        d_hidden_state,
        hidden_size);
    CUDA_CHECK(cudaGetLastError());

    host_output.resize(hidden_size);
    CUDA_CHECK(cudaMemcpy(host_output.data(),
                          d_hidden_state,
                          hidden_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void Inference::run_self_attention(
    const std::vector<float>& host_input,
    std::vector<float>& host_output,
    std::vector<float>& host_attention_weights,
    float* execution_time_ms)
{
    if (!attention_weights_loaded)
    {
        throw std::runtime_error(
            "attention weights must be loaded before attention");
    }
    validate_input(host_input);

    const size_t input_count =
        static_cast<size_t>(time_steps) * input_size;
    CUDA_CHECK(cudaMemcpy(d_input,
                          host_input.data(),
                          input_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    CudaEventTimer timer;
    timer.start();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // The time-major host input is also a column-major [input_size, T]
    // matrix. The output has the same dual view: column-major [H, T] and
    // time-major [T, H].
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             d_model,
                             time_steps,
                             input_size,
                             &alpha,
                             d_Wq,
                             d_model,
                             d_input,
                             input_size,
                             &beta,
                             d_Q,
                             d_model));
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             d_model,
                             time_steps,
                             input_size,
                             &alpha,
                             d_Wk,
                             d_model,
                             d_input,
                             input_size,
                             &beta,
                             d_K,
                             d_model));
    CUBLAS_CHECK(cublasSgemm(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             d_model,
                             time_steps,
                             input_size,
                             &alpha,
                             d_Wv,
                             d_model,
                             d_input,
                             input_size,
                             &beta,
                             d_V,
                             d_model));

    constexpr int threads = 256;
    attention_scores_kernel<<<blocks_for(time_steps, threads), threads>>>(
        d_Q,
        d_K,
        d_attention_scores,
        time_steps,
        d_model,
        time_steps - 1);
    CUDA_CHECK(cudaGetLastError());

    const int softmax_threads = next_power_of_two(time_steps);
    attention_softmax_kernel<<<1,
                               softmax_threads,
                               softmax_threads * sizeof(float)>>>(
        d_attention_scores, time_steps);
    CUDA_CHECK(cudaGetLastError());

    attention_weighted_sum_kernel<<<blocks_for(d_model, threads), threads>>>(
        d_V,
        d_attention_scores,
        d_attention_output,
        time_steps,
        d_model);
    CUDA_CHECK(cudaGetLastError());

    const float elapsed_ms = timer.stop();
    if (execution_time_ms != nullptr)
    {
        *execution_time_ms = elapsed_ms;
    }

    host_output.resize(d_model);
    host_attention_weights.resize(time_steps);
    CUDA_CHECK(cudaMemcpy(host_output.data(),
                          d_attention_output,
                          d_model * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_attention_weights.data(),
                          d_attention_scores,
                          time_steps * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

std::string Inference::device_name() const
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}
