#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "inference.h"
#include "cpu_reference.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(cuda_lstm_ops, m) {
    m.doc() = "cuda lstm and self attention inference engine";

    py::class_<Inference>(m, "Inference")
        .def(py::init<int, int, int>(), py::arg("input_dim"), py::arg("hidden_dim"), py::arg("seq_len"))
        .def("run_inference", [](Inference& self, const vector<float>& input, const vector<float>& weights, const vector<float>& biases) {
            vector<float> output;
            float time_ms = 0.0f;
            self.run_inference(input, weights, biases, output, &time_ms);
            return py::make_tuple(output, time_ms);
        }, "run lstm inference", py::arg("input"), py::arg("weights"), py::arg("biases"))
        .def("run_inference_fp16", [](Inference& self, const vector<float>& input, const vector<float>& weights, const vector<float>& biases) {
            vector<float> output;
            float time_ms = 0.0f;
            self.run_inference_fp16(input, weights, biases, output, &time_ms);
            return py::make_tuple(output, time_ms);
        }, "run lstm inference in fp16", py::arg("input"), py::arg("weights"), py::arg("biases"))
        .def("run_self_attention", [](Inference& self, const vector<float>& input, const vector<float>& Wq, const vector<float>& Wk, const vector<float>& Wv) {
            vector<float> output;
            self.run_self_attention(input, Wq, Wk, Wv, output);
            return output;
        }, "run self attention", py::arg("input"), py::arg("Wq"), py::arg("Wk"), py::arg("Wv"));

    py::class_<CpuInference>(m, "CpuInference")
        .def(py::init<int, int, int>(), py::arg("input_dim"), py::arg("hidden_dim"), py::arg("seq_len"))
        .def("run_inference", [](CpuInference& self, const vector<float>& input, const vector<float>& weights, const vector<float>& biases) {
            vector<float> output;
            float time_ms = 0.0f;
            self.run_inference(input, weights, biases, output, &time_ms);
            return py::make_tuple(output, time_ms);
        }, "run cpu lstm inference", py::arg("input"), py::arg("weights"), py::arg("biases"));
}