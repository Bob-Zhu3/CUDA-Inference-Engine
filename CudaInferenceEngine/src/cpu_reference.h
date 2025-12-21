#pragma once
#include <vector>
#include <cmath>

using namespace std;

class CpuInference
{
public:
    CpuInference(int input_dim, int hidden_dim, int seq_len);
    ~CpuInference();

    void run_inference(const vector<float>& input,
                       const vector<float>& weights,
                       const vector<float>& biases,
                       vector<float>& output,
                       float* execution_time_ms = nullptr);

private:
    int input_size;
    int hidden_size;
    int time_steps;

    vector<float> hidden_state;
    vector<float> cell_state;

    float sigmoid(float x);
    float tanh_custom(float x);
};

