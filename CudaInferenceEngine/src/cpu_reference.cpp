#include "cpu_reference.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace std;

CpuInference::CpuInference(int in_dim, int hid_dim, int seq_len)
    : input_size(in_dim), hidden_size(hid_dim), time_steps(seq_len)
{
    hidden_state.resize(hidden_size, 0.0f);
    cell_state.resize(hidden_size, 0.0f);
}

CpuInference::~CpuInference() {}

float CpuInference::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

float CpuInference::tanh_custom(float x)
{
    return std::tanh(x);
}

void CpuInference::run_inference(const vector<float>& input,
                                 const vector<float>& weights,
                                 const vector<float>& biases,
                                 vector<float>& output,
                                 float* execution_time_ms)
{
    std::fill(hidden_state.begin(), hidden_state.end(), 0.0f);
    std::fill(cell_state.begin(), cell_state.end(), 0.0f);

    vector<float> z(4 * hidden_size);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < time_steps; t++)
    {
        for (int i = 0; i < 4 * hidden_size; i++)
        {
            float dot = 0.0f;
            for (int j = 0; j < input_size; j++)
            {
                float w_val = weights[i + j * 4 * hidden_size];
                float x_val = input[t * input_size + j];
                dot += w_val * x_val;
            }
            z[i] = dot + biases[i];
        }

        for (int h = 0; h < hidden_size; h++)
        {
            float f_t = sigmoid(z[h]);
            float i_t = sigmoid(z[h + hidden_size]);
            float c_tilde_t = tanh_custom(z[h + 2 * hidden_size]);
            float o_t = sigmoid(z[h + 3 * hidden_size]);

            cell_state[h] = f_t * cell_state[h] + i_t * c_tilde_t;
            hidden_state[h] = o_t * tanh_custom(cell_state[h]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    if (execution_time_ms != nullptr)
    {
        *execution_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    }

    output = hidden_state;
}