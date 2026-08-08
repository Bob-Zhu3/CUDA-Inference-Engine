#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <vector>

#include "cpu_reference.h"
#include "inference.h"

namespace py = pybind11;

using FloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace
{

std::vector<float> array_to_vector(const FloatArray& array)
{
    const py::buffer_info info = array.request();
    const auto* data = static_cast<const float*>(info.ptr);
    return std::vector<float>(data, data + info.size);
}

py::array_t<float> vector_to_array(const std::vector<float>& values)
{
    py::array_t<float> result(values.size());
    std::memcpy(result.mutable_data(),
                values.data(),
                values.size() * sizeof(float));
    return result;
}

} // namespace

PYBIND11_MODULE(cuda_lstm_ops, module)
{
    module.doc() =
        "Single-layer CUDA LSTM and last-query self-attention engine";

    py::class_<Inference>(module, "Inference")
        .def(py::init<int, int, int>(),
             py::arg("input_dim"),
             py::arg("hidden_dim"),
             py::arg("seq_len"))
        .def(
            "load_lstm_weights",
            [](Inference& self,
               const FloatArray& input_weights,
               const FloatArray& recurrent_weights,
               const FloatArray& biases)
            {
                const std::vector<float> input_values =
                    array_to_vector(input_weights);
                const std::vector<float> recurrent_values =
                    array_to_vector(recurrent_weights);
                const std::vector<float> bias_values =
                    array_to_vector(biases);

                py::gil_scoped_release release;
                self.load_lstm_weights(
                    input_values, recurrent_values, bias_values);
            },
            py::arg("input_weights"),
            py::arg("recurrent_weights"),
            py::arg("biases"),
            "Load input-major LSTM weights once and retain them on the GPU")
        .def(
            "load_attention_weights",
            [](Inference& self,
               const FloatArray& Wq,
               const FloatArray& Wk,
               const FloatArray& Wv)
            {
                const std::vector<float> q_values = array_to_vector(Wq);
                const std::vector<float> k_values = array_to_vector(Wk);
                const std::vector<float> v_values = array_to_vector(Wv);

                py::gil_scoped_release release;
                self.load_attention_weights(q_values, k_values, v_values);
            },
            py::arg("Wq"),
            py::arg("Wk"),
            py::arg("Wv"),
            "Load input-major query, key, and value weights")
        .def(
            "run_inference",
            [](Inference& self, const FloatArray& input)
            {
                const std::vector<float> input_values =
                    array_to_vector(input);
                std::vector<float> output;
                float execution_time_ms = 0.0f;

                {
                    py::gil_scoped_release release;
                    self.run_inference(
                        input_values, output, &execution_time_ms);
                }

                return py::make_tuple(
                    vector_to_array(output), execution_time_ms);
            },
            py::arg("input"),
            "Run regular-precision LSTM inference")
        .def(
            "run_inference_fp16",
            [](Inference& self, const FloatArray& input)
            {
                const std::vector<float> input_values =
                    array_to_vector(input);
                std::vector<float> output;
                float execution_time_ms = 0.0f;

                {
                    py::gil_scoped_release release;
                    self.run_inference_fp16(
                        input_values, output, &execution_time_ms);
                }

                return py::make_tuple(
                    vector_to_array(output), execution_time_ms);
            },
            py::arg("input"),
            "Run mixed-FP16 LSTM inference")
        .def(
            "run_self_attention",
            [](Inference& self, const FloatArray& input)
            {
                const std::vector<float> input_values =
                    array_to_vector(input);
                std::vector<float> output;
                std::vector<float> weights;
                float execution_time_ms = 0.0f;

                {
                    py::gil_scoped_release release;
                    self.run_self_attention(
                        input_values,
                        output,
                        weights,
                        &execution_time_ms);
                }

                return py::make_tuple(vector_to_array(output),
                                      vector_to_array(weights),
                                      execution_time_ms);
            },
            py::arg("input"),
            "Run last-query self-attention and return output, weights, time")
        .def("device_name", &Inference::device_name);

    py::class_<CpuInference>(module, "CpuInference")
        .def(py::init<int, int, int>(),
             py::arg("input_dim"),
             py::arg("hidden_dim"),
             py::arg("seq_len"))
        .def(
            "load_lstm_weights",
            [](CpuInference& self,
               const FloatArray& input_weights,
               const FloatArray& recurrent_weights,
               const FloatArray& biases)
            {
                const std::vector<float> input_values =
                    array_to_vector(input_weights);
                const std::vector<float> recurrent_values =
                    array_to_vector(recurrent_weights);
                const std::vector<float> bias_values =
                    array_to_vector(biases);

                py::gil_scoped_release release;
                self.load_lstm_weights(
                    input_values, recurrent_values, bias_values);
            },
            py::arg("input_weights"),
            py::arg("recurrent_weights"),
            py::arg("biases"))
        .def(
            "run_inference",
            [](const CpuInference& self, const FloatArray& input)
            {
                const std::vector<float> input_values =
                    array_to_vector(input);
                std::vector<float> output;
                float execution_time_ms = 0.0f;

                {
                    py::gil_scoped_release release;
                    self.run_inference(
                        input_values, output, &execution_time_ms);
                }

                return py::make_tuple(
                    vector_to_array(output), execution_time_ms);
            },
            py::arg("input"),
            "Run the single-threaded CPU reference");
}
