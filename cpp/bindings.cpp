#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "indicators.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fastindicators, m) {
    m.def("sma", &sma, "Simple Moving Average");
    m.def("rsi", &rsi, "Relative Strength Index");
}
