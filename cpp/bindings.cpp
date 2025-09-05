// cpp/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "execution_engine.h"  // assume this declares ExecutionEngine class

namespace py = pybind11;

// create a singleton engine instance
static std::unique_ptr<ExecutionEngine> g_engine;

PYBIND11_MODULE(execution_engine_cpp, m) {
    m.doc() = "Execution engine bindings";

    // init singleton on import
    if (!g_engine) {
        g_engine = std::make_unique<ExecutionEngine>();
    }

    // Expose a minimal ExecutionEngine class (optional)
    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("reset", &ExecutionEngine::reset)
        .def("place_order", &ExecutionEngine::place_order,
             py::arg("symbol"), py::arg("qty"), py::arg("price"),
             py::arg("strike"), py::arg("sigma"), py::arg("is_call"), py::arg("expiry_days"))
        .def("account_status", &ExecutionEngine::account_status, py::arg("spot"))
        .def("portfolio_greeks", &ExecutionEngine::portfolio_greeks, py::arg("spot"))
        .def("get_positions", &ExecutionEngine::get_positions)
        .def("get_trade_log", &ExecutionEngine::get_trade_log);

    // Export a singleton instance and free functions that call through to it.
    m.attr("engine") = py::cast(g_engine.get(), py::return_value_policy::reference);

    m.def("reset_engine", [](double init_balance) {
        g_engine->reset(init_balance);
    }, py::arg("init_balance"));

    m.def("place_order", [](const std::string &symbol, int qty, double price,
                           double strike, double sigma, bool is_call, int expiry_days) {
        g_engine->place_order(symbol, qty, price, strike, sigma, is_call, expiry_days);
    }, py::arg("symbol"), py::arg("qty"), py::arg("price"),
       py::arg("strike"), py::arg("sigma"), py::arg("is_call"), py::arg("expiry_days"));

    m.def("account_status", [](double spot) {
        return g_engine->account_status(spot);
    }, py::arg("spot"));

    m.def("portfolio_greeks", [](double spot) {
        // Ensure the C++ method returns a tuple/datastructure compatible with pybind
        return g_engine->portfolio_greeks(spot);  // must be std::tuple<double,double,double,double>
    }, py::arg("spot"));

    m.def("get_positions", []() {
        return g_engine->get_positions();
    });

    m.def("get_trade_log", []() {
        return g_engine->get_trade_log();
    });
}
