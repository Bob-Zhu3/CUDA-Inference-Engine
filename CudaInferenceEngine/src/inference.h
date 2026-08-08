#pragma once

#include <cublas_v2.h>
#include <string>
#include <vector>

class Inference
{
public:
    Inference(int input_dim, int hidden_dim, int seq_len);
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;
    Inference(Inference&&) = delete;
    Inference& operator=(Inference&&) = delete;

    // Weight layout is input-major so a C-contiguous NumPy array with shape
    // [input_size, 4 * hidden_size] can be copied directly to cuBLAS.
    void load_lstm_weights(const std::vector<float>& input_weights,
                           const std::vector<float>& recurrent_weights,
                           const std::vector<float>& biases);

    // Each attention matrix is input-major with shape [input_size, hidden_size].
    void load_attention_weights(const std::vector<float>& Wq,
                                const std::vector<float>& Wk,
                                const std::vector<float>& Wv);

    void run_inference(const std::vector<float>& input,
                       std::vector<float>& output,
                       float* execution_time_ms = nullptr);

    void run_inference_fp16(const std::vector<float>& input,
                            std::vector<float>& output,
                            float* execution_time_ms = nullptr);

    // Computes single-head, last-query self-attention. The returned weights
    // have one value per time step and sum to approximately one.
    void run_self_attention(const std::vector<float>& input,
                            std::vector<float>& output,
                            std::vector<float>& attention_weights,
                            float* execution_time_ms = nullptr);

    std::string device_name() const;

private:
    int input_size;
    int hidden_size;
    int time_steps;
    int d_model;

    bool lstm_weights_loaded;
    bool attention_weights_loaded;

    float* d_input;
    float* d_input_weights;
    float* d_recurrent_weights;
    float* d_biases;
    float* d_hidden_state;
    float* d_cell_state;
    float* d_z;

    float* d_Wq;
    float* d_Wk;
    float* d_Wv;
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_attention_scores;
    float* d_attention_output;

    // Stored as void* so callers of the header do not need cuda_fp16.h.
    void* d_input_fp16;
    void* d_input_weights_fp16;
    void* d_recurrent_weights_fp16;
    void* d_biases_fp16;
    void* d_hidden_state_fp16;
    void* d_cell_state_fp16;
    void* d_z_fp16;

    cublasHandle_t cublas_handle;

    void allocate_memory();
    void release_memory() noexcept;
    void validate_input(const std::vector<float>& input) const;
};
