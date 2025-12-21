#include "inference.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>

using namespace std;

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh2(float x)
{
    return tanhf(x);
}

__global__ void lstm_ew_kernel(float *z, float *cell_state, float *hidden_state, int hidden_size)
{
    int idx = threadIdx.x;
    if (idx >= hidden_size)
        return;
    float f_t = sigmoid(z[idx]);
    float i_t = sigmoid(z[idx + hidden_size]);
    float c_tilde_t = tanh2(z[idx + 2 * hidden_size]);
    float o_t = sigmoid(z[idx + 3 * hidden_size]);
    cell_state[idx] = f_t * cell_state[idx] + i_t * c_tilde_t;
    hidden_state[idx] = o_t * tanh2(cell_state[idx]);
}

__global__ void float2half_kernel(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void half2float_kernel(const __half* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void lstm_ew_kernel_fp16(__half *z, __half *cell_state, __half *hidden_state, int hidden_size)
{
    int idx = threadIdx.x;
    if (idx >= hidden_size)
        return;
    
    float z_f = __half2float(z[idx]);
    float z_i = __half2float(z[idx + hidden_size]);
    float z_c = __half2float(z[idx + 2 * hidden_size]);
    float z_o = __half2float(z[idx + 3 * hidden_size]);
    
    float f_t = sigmoid(z_f);
    float i_t = sigmoid(z_i);
    float c_tilde_t = tanh2(z_c);
    float o_t = sigmoid(z_o);
    
    float c_prev = __half2float(cell_state[idx]);
    float c_new = f_t * c_prev + i_t * c_tilde_t;
    float h_new = o_t * tanh2(c_new);
    
    cell_state[idx] = __float2half(c_new);
    hidden_state[idx] = __float2half(h_new);
}

__global__ void attention_scores(const float *Q, const float *K, float *scores, int T, int d_model, int query_index)
{
    int i = threadIdx.x;
    if (i >= T)
        return;
    const float *q = Q + query_index * d_model;
    const float *k_i = K + i * d_model;
    float dot = 0.f;
    for (int j = 0; j < d_model; j++)
        dot += q[j] * k_i[j];
    scores[i] = dot / sqrtf((float)d_model);
}

__global__ void attention_weighted_sum(const float *V, const float *weights, float *output, int T, int d_model)
{
    int j = threadIdx.x;
    if (j >= d_model)
        return;
    float sum = 0.f;
    for (int i = 0; i < T; i++)
        sum += weights[i] * V[i * d_model + j];
    output[j] = sum;
}

Inference::Inference(int in_dim, int hid_dim, int seq_len)
    : input_size(in_dim), hidden_size(hid_dim), time_steps(seq_len), d_model(hid_dim)
{
    cublasCreate(&cublas_handle);
    allocate_memory();

    // For warm up purposes
    vector<float> dummy_in(time_steps * input_size, 0.1f);
    vector<float> dummy_w(4 * hidden_size * input_size, 0.1f);
    vector<float> dummy_b(4 * hidden_size, 0.1f);
    vector<float> dummy_out;
    
    run_inference(dummy_in, dummy_w, dummy_b, dummy_out);
    
    run_inference_fp16(dummy_in, dummy_w, dummy_b, dummy_out);
    
    cudaDeviceSynchronize(); 
}

Inference::~Inference()
{
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_hidden_state);
    cudaFree(d_cell_state);
    cudaFree(d_z);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);

    cudaFree(d_input_fp16);
    cudaFree(d_weights_fp16);
    cudaFree(d_biases_fp16);
    cudaFree(d_hidden_state_fp16);
    cudaFree(d_cell_state_fp16);
    cudaFree(d_z_fp16);

    cublasDestroy(cublas_handle);
}

void Inference::allocate_memory()
{
    cudaMalloc((void **)&d_input, time_steps * input_size * sizeof(float));
    cudaMalloc((void **)&d_weights, 4 * hidden_size * input_size * sizeof(float));
    cudaMalloc((void **)&d_biases, 4 * hidden_size * sizeof(float));
    cudaMalloc((void **)&d_hidden_state, hidden_size * sizeof(float));
    cudaMalloc((void **)&d_cell_state, hidden_size * sizeof(float));

    cudaMalloc((void **)&d_z, 4 * hidden_size * sizeof(float));

    cudaMalloc((void **)&d_Wq, input_size * d_model * sizeof(float));
    cudaMalloc((void **)&d_Wk, input_size * d_model * sizeof(float));
    cudaMalloc((void **)&d_Wv, input_size * d_model * sizeof(float));

    cudaMemset(d_hidden_state, 0, hidden_size * sizeof(float));
    cudaMemset(d_cell_state, 0, hidden_size * sizeof(float));

    // FP16 allocations
    cudaMalloc((void **)&d_input_fp16, time_steps * input_size * sizeof(__half));
    cudaMalloc((void **)&d_weights_fp16, 4 * hidden_size * input_size * sizeof(__half));
    cudaMalloc((void **)&d_biases_fp16, 4 * hidden_size * sizeof(__half));
    cudaMalloc((void **)&d_hidden_state_fp16, hidden_size * sizeof(__half));
    cudaMalloc((void **)&d_cell_state_fp16, hidden_size * sizeof(__half));
    cudaMalloc((void **)&d_z_fp16, 4 * hidden_size * sizeof(__half));

    cudaMemset(d_hidden_state_fp16, 0, hidden_size * sizeof(__half));
    cudaMemset(d_cell_state_fp16, 0, hidden_size * sizeof(__half));
}

void Inference::run_inference(const vector<float>& h_input,
                               const vector<float>& h_weights,
                               const vector<float>& h_biases,
                               vector<float>& h_output,
                               float* execution_time_ms)
{
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), h_biases.size() * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int t = 0; t < time_steps; t++)
    {
        float *d_input_t = d_input + t * input_size;
        cudaMemcpy(d_z, d_biases, 4 * hidden_size * sizeof(float), cudaMemcpyDeviceToDevice);
        float beta = 1.0f;
        cublasSgemv(cublas_handle, CUBLAS_OP_N, 4 * hidden_size, input_size, &alpha, d_weights, 4 * hidden_size, d_input_t, 1, &beta, d_z, 1);
        lstm_ew_kernel<<<1, hidden_size>>>(d_z, d_cell_state, d_hidden_state, hidden_size);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (execution_time_ms != nullptr) {
        cudaEventElapsedTime(execution_time_ms, start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    h_output.resize(hidden_size);
    cudaMemcpy(h_output.data(), d_hidden_state, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Inference::run_inference_fp16(const vector<float>& h_input,
                                   const vector<float>& h_weights,
                                   const vector<float>& h_biases,
                                   vector<float>& h_output,
                                   float* execution_time_ms)
{
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases.data(), h_biases.size() * sizeof(float), cudaMemcpyHostToDevice);

    int total_input_elements = time_steps * input_size;
    int total_weight_elements = 4 * hidden_size * input_size;
    int total_bias_elements = 4 * hidden_size;

    int threads = 256;
    int blocks_input = (total_input_elements + threads - 1) / threads;
    int blocks_weights = (total_weight_elements + threads - 1) / threads;
    int blocks_biases = (total_bias_elements + threads - 1) / threads;

    float2half_kernel<<<blocks_input, threads>>>(d_input, (__half*)d_input_fp16, total_input_elements);
    float2half_kernel<<<blocks_weights, threads>>>(d_weights, (__half*)d_weights_fp16, total_weight_elements);
    float2half_kernel<<<blocks_biases, threads>>>(d_biases, (__half*)d_biases_fp16, total_bias_elements);

    cudaMemset(d_hidden_state_fp16, 0, hidden_size * sizeof(__half));
    cudaMemset(d_cell_state_fp16, 0, hidden_size * sizeof(__half));

    float alpha_f = 1.0f;
    float beta_f = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int t = 0; t < time_steps; t++)
    {
        __half *d_input_t_fp16 = (__half*)d_input_fp16 + t * input_size;
        
        cudaMemcpy(d_z_fp16, d_biases_fp16, 4 * hidden_size * sizeof(__half), cudaMemcpyDeviceToDevice);
        
        cublasGemmEx(cublas_handle, 
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     4 * hidden_size, 1, input_size,
                     &alpha_f,
                     d_weights_fp16, CUDA_R_16F, 4 * hidden_size,
                     d_input_t_fp16, CUDA_R_16F, input_size,
                     &beta_f,
                     d_z_fp16, CUDA_R_16F, 4 * hidden_size,
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        lstm_ew_kernel_fp16<<<1, hidden_size>>>((__half*)d_z_fp16, (__half*)d_cell_state_fp16, (__half*)d_hidden_state_fp16, hidden_size);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    if (execution_time_ms != nullptr) {
        cudaEventElapsedTime(execution_time_ms, start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    h_output.resize(hidden_size);
    int blocks_output = (hidden_size + threads - 1) / threads;
    half2float_kernel<<<blocks_output, threads>>>((__half*)d_hidden_state_fp16, d_hidden_state, hidden_size);
    
    cudaMemcpy(h_output.data(), d_hidden_state, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void Inference::run_self_attention(const vector<float>& h_input,
                                    const vector<float>& h_Wq,
                                    const vector<float>& h_Wk,
                                    const vector<float>& h_Wv,
                                    vector<float>& h_output)
{
    cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_Q;
    float *d_K;
    float *d_V;
    cudaMalloc((void **)&d_Q, time_steps * d_model * sizeof(float));
    cudaMalloc((void **)&d_K, time_steps * d_model * sizeof(float));
    cudaMalloc((void **)&d_V, time_steps * d_model * sizeof(float));
    float *d_scores;
    float *d_output;
    cudaMalloc((void **)&d_scores, time_steps * sizeof(float));
    cudaMalloc((void **)&d_output, d_model * sizeof(float));
    for (int t = 0; t < time_steps; t++)
    {
        float *d_input_t = d_input + t * input_size;
        float *d_Q_t = d_Q + t * d_model;
        float *d_K_t = d_K + t * d_model;
        float *d_V_t = d_V + t * d_model;
        cublasSgemv(cublas_handle, CUBLAS_OP_N, d_model, input_size, &alpha, d_Wq, d_model, d_input_t, 1, &beta, d_Q_t, 1);
        cublasSgemv(cublas_handle, CUBLAS_OP_N, d_model, input_size, &alpha, d_Wk, d_model, d_input_t, 1, &beta, d_K_t, 1);
        cublasSgemv(cublas_handle, CUBLAS_OP_N, d_model, input_size, &alpha, d_Wv, d_model, d_input_t, 1, &beta, d_V_t, 1);
    }
    int query_index = time_steps - 1;
    attention_scores<<<1, time_steps>>>(d_Q, d_K, d_scores, time_steps, d_model, query_index);
    cudaDeviceSynchronize();
    vector<float> h_scores(time_steps);
    cudaMemcpy(h_scores.data(), d_scores, time_steps * sizeof(float), cudaMemcpyDeviceToHost);
    float max_score = h_scores[0];
    for (int i = 1; i < time_steps; i++)
        if (h_scores[i] > max_score) max_score = h_scores[i];
    float sum_exp = 0.f;
    for (int i = 0; i < time_steps; i++)
    {
        h_scores[i] = expf(h_scores[i] - max_score);
        sum_exp += h_scores[i];
    }
    for (int i = 0; i < time_steps; i++)
        h_scores[i] /= sum_exp;
    cudaMemcpy(d_scores, h_scores.data(), time_steps * sizeof(float), cudaMemcpyHostToDevice);
    attention_weighted_sum<<<1, d_model>>>(d_V, d_scores, d_output, time_steps, d_model);
    cudaDeviceSynchronize();
    h_output.resize(d_model);
    cudaMemcpy(h_output.data(), d_output, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_output);
}