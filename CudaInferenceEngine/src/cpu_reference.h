#pragma once

#include <vector>

class CpuInference
{
public:
    CpuInference(int input_dim, int hidden_dim, int seq_len);

    void load_lstm_weights(const std::vector<float>& input_weights,
                           const std::vector<float>& recurrent_weights,
                           const std::vector<float>& biases);

    void run_inference(const std::vector<float>& input,
                       std::vector<float>& output,
                       float* execution_time_ms = nullptr) const;

private:
    int input_size;
    int hidden_size;
    int time_steps;
    bool weights_loaded;

    std::vector<float> input_weights;
    std::vector<float> recurrent_weights;
    std::vector<float> biases;

    static float sigmoid(float x);
};
