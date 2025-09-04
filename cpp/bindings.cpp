#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "execution_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(execution_engine_cpp, m) {
    m.doc() = "C++ ExecutionEngine (premium-based PnL, FIFO realized, per-contract Greeks)";

    py::class_<Position>(m, "Position")
        .def_readonly("symbol",       &Position::symbol)
        .def_readonly("qty",          &Position::qty)
        .def_readonly("strike",       &Position::strike)
        .def_readonly("sigma",        &Position::sigma)
        .def_readonly("is_call",      &Position::is_call)
        .def_readonly("expiry_days",  &Position::expiry_days)
        .def_readonly("entry_price",  &Position::entry_price)
        .def_readonly("realized_pnl", &Position::realized_pnl)
        .def_readonly("unrealized_pnl",&Position::unrealized_pnl);

    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("reset",                 &ExecutionEngine::reset)
        .def("place_order",           &ExecutionEngine::place_order)
        .def("account_status",        &ExecutionEngine::account_status)
        .def("portfolio_greeks",      &ExecutionEngine::portfolio_greeks)
        .def("get_positions",         &ExecutionEngine::get_positions,
                                       py::return_value_policy::reference_internal)
        .def("get_positions_with_pnl",&ExecutionEngine::get_positions_with_pnl);
}
