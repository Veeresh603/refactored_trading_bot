#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;
namespace py = pybind11;

struct Trade {
    string type;
    double entry_price;
    double exit_price;
    double pnl;
};

// Simple backtest: Moving Average crossover strategy
vector<Trade> backtest(
    const vector<double>& close,
    const vector<int>& signals // 1=BUY, -1=SELL, 0=HOLD
) {
    vector<Trade> trades;
    int position = 0;
    double entry_price = 0;

    for (size_t i = 0; i < close.size(); i++) {
        if (signals[i] == 1 && position == 0) {
            // BUY
            position = 1;
            entry_price = close[i];
        }
        else if (signals[i] == -1 && position == 1) {
            // SELL
            double pnl = close[i] - entry_price;
            trades.push_back({"BUY->SELL", entry_price, close[i], pnl});
            position = 0;
        }
    }

    return trades;
}

PYBIND11_MODULE(fastbt, m) {
    py::class_<Trade>(m, "Trade")
        .def_readonly("type", &Trade::type)
        .def_readonly("entry_price", &Trade::entry_price)
        .def_readonly("exit_price", &Trade::exit_price)
        .def_readonly("pnl", &Trade::pnl);

    m.def("backtest", &backtest, "Run fast backtest");
}
