#include "cpu_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

CpuInference::CpuInference(int in_dim, int hid_dim, int seq_len)
    : input_size(in_dim),
      hidden_size(hid_dim),
      time_steps(seq_len),
      weights_loaded(false)
{
    if (input_size <= 0 || hidden_size <= 0 || time_steps <= 0)
    {
        throw std::invalid_argument("input_dim, hidden_dim, and seq_len must all be positive");
    }
}

float CpuInference::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void CpuInference::load_lstm_weights(const std::vector<float>& new_input_weights,
                                     const std::vector<float>& new_recurrent_weights,
                                     const std::vector<float>& new_biases)
{
    const size_t gate_size = static_cast<size_t>(4) * hidden_size;
    const size_t expected_input_weights = gate_size * input_size;
    const size_t expected_recurrent_weights = gate_size * hidden_size;

    if (new_input_weights.size() != expected_input_weights)
    {
        throw std::invalid_argument("input_weights must contain input_size * 4 * hidden_size values");
    }
    if (new_recurrent_weights.size() != expected_recurrent_weights)
    {
        throw std::invalid_argument("recurrent_weights must contain hidden_size * 4 * hidden_size values");
    }
    if (new_biases.size() != gate_size)
    {
        throw std::invalid_argument("biases must contain 4 * hidden_size values");
    }

    input_weights = new_input_weights;
    recurrent_weights = new_recurrent_weights;
    biases = new_biases;
    weights_loaded = true;
}

void CpuInference::run_inference(const std::vector<float>& input,
                                 std::vector<float>& output,
                                 float* execution_time_ms) const
{
    if (!weights_loaded)
    {
        throw std::runtime_error("LSTM weights must be loaded before inference");
    }

    const size_t expected_input = static_cast<size_t>(time_steps) * input_size;
    if (input.size() != expected_input)
    {
        throw std::invalid_argument("input must contain seq_len * input_size values");
    }

    std::vector<float> hidden_state(hidden_size, 0.0f);
    std::vector<float> cell_state(hidden_size, 0.0f);
    std::vector<float> z(static_cast<size_t>(4) * hidden_size);

    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < time_steps; ++t)
    {
        std::copy(biases.begin(), biases.end(), z.begin());

        // The arrays are input-major. For every input value, all 4H output
        // weights are contiguous. This matches the column-major view used by
        // cuBLAS in the GPU implementation.
        for (int input_index = 0; input_index < input_size; ++input_index)
        {
            const float x = input[static_cast<size_t>(t) * input_size + input_index];
            const size_t weight_base =
                static_cast<size_t>(input_index) * 4 * hidden_size;

            for (int gate_index = 0; gate_index < 4 * hidden_size; ++gate_index)
            {
                z[gate_index] += input_weights[weight_base + gate_index] * x;
            }
        }

        for (int previous_hidden_index = 0;
             previous_hidden_index < hidden_size;
             ++previous_hidden_index)
        {
            const float previous_hidden = hidden_state[previous_hidden_index];
            const size_t weight_base =
                static_cast<size_t>(previous_hidden_index) * 4 * hidden_size;

            for (int gate_index = 0; gate_index < 4 * hidden_size; ++gate_index)
            {
                z[gate_index] +=
                    recurrent_weights[weight_base + gate_index] * previous_hidden;
            }
        }

        for (int h = 0; h < hidden_size; ++h)
        {
            const float forget_gate = sigmoid(z[h]);
            const float input_gate = sigmoid(z[h + hidden_size]);
            const float candidate = std::tanh(z[h + 2 * hidden_size]);
            const float output_gate = sigmoid(z[h + 3 * hidden_size]);

            cell_state[h] =
                forget_gate * cell_state[h] + input_gate * candidate;
            hidden_state[h] = output_gate * std::tanh(cell_state[h]);
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    if (execution_time_ms != nullptr)
    {
        *execution_time_ms =
            std::chrono::duration<float, std::milli>(end - start).count();
    }

    output = hidden_state;
}
