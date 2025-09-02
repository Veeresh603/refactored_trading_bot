#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include headers
#include "indicators.h"
// Add more headers later: execution_engine.h, risk_manager.h, backtester.h

namespace py = pybind11;

PYBIND11_MODULE(execution_engine, m) {
    m.doc() = "C++ trading engine bindings";

    // Indicators
    m.def("rsi", &rsi, "Relative Strength Index",
          py::arg("prices"), py::arg("period"));
    m.def("sma", &sma, "Simple Moving Average",
          py::arg("prices"), py::arg("period"));

    // TODO: add bindings for execution_engine, risk_manager, backtester
}
