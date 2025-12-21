#pragma once
#include <cublas_v2.h>
#include <vector>

using namespace std;

class Inference
{
public:
    Inference(int input_dim, int hidden_dim, int seq_len);

    ~Inference();

    void run_inference(const vector<float>& h_input,
                       const vector<float>& h_weights,
                       const vector<float>& h_biases,
                       vector<float>& h_output,
                       float* execution_time_ms = nullptr);

    void run_inference_fp16(const vector<float>& h_input,
                            const vector<float>& h_weights,
                            const vector<float>& h_biases,
                            vector<float>& h_output,
                            float* execution_time_ms = nullptr);

    void run_self_attention(const vector<float>& h_input,
                            const vector<float>& h_Wq,
                            const vector<float>& h_Wk,
                            const vector<float>& h_Wv,
                            vector<float>& h_output);

private:
    int input_size;
    int hidden_size;
    int time_steps;
    int d_model;

    float *d_input, *d_weights, *d_biases;
    float *d_hidden_state, *d_cell_state;
    float *d_z;

    float *d_Wq, *d_Wk, *d_Wv;

    cublasHandle_t cublas_handle;

    // fp16 buffers (void* to avoid including cuda_fp16.h in header)
    void *d_input_fp16, *d_weights_fp16, *d_biases_fp16;
    void *d_hidden_state_fp16, *d_cell_state_fp16;
    void *d_z_fp16;

    void allocate_memory();
};