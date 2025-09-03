#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "execution_engine.h"

namespace py = pybind11;

// Expose ExecutionEngine singleton instance to Python
PYBIND11_MODULE(execution_cpp, m) {
    m.doc() = "C++ Execution Engine for Trading (PyBind11)";

    // Place Order
    m.def("place_order", [](const std::string& symbol, int qty, double price,
                            double strike, double sigma, bool is_call, double expiry_days) {
        engine.place_order(symbol, qty, price, strike, sigma, is_call, expiry_days);
    });

    // Account Status
    m.def("account_status", [](double spot) {
        return engine.account_status(spot);
    });

    // Portfolio Greeks
    m.def("portfolio_greeks", [](double spot) {
        auto [delta, gamma, vega, theta] = engine.portfolio_greeks(spot);
        return std::unordered_map<std::string,double>{
            {"delta", delta},
            {"gamma", gamma},
            {"vega", vega},
            {"theta", theta}
        };
    });

    // Trade Log
    m.def("get_trade_log", []() {
        return engine.get_trade_log();
    });

    // Reset
    m.def("reset_engine", [](double balance) {
        engine.reset(balance);
    });
}
