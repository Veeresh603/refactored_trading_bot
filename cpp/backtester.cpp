#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "backtester.h"

namespace py = pybind11;

struct Trade {
    std::string symbol;
    std::string type;   // BUY/SELL
    double entry_price;
    double exit_price;
    double pnl;
    double strike;
    bool is_call;
    double expiry_days;
};

// -----------------------------
// Black-Scholes Pricing
// -----------------------------
static double option_price(double spot, double strike, double sigma, double t, bool is_call) {
    if (t <= 0) return std::max((is_call ? spot - strike : strike - spot), 0.0);

    double d1 = (std::log(spot / strike) + (0.5 * sigma * sigma) * t) / (sigma * std::sqrt(t));
    double d2 = d1 - sigma * std::sqrt(t);
    double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

    if (is_call)
        return spot * Nd1 - strike * Nd2;
    else
        return strike * (1.0 - Nd2) - spot * (1.0 - Nd1);
}

// -----------------------------
// Options Backtester
// -----------------------------
std::unordered_map<std::string, std::vector<double>> backtest_options(
    const std::vector<double>& spot_prices,
    const std::vector<int>& signals,    // 1=BUY CALL, -1=BUY PUT, 0=HOLD
    double strike,
    double sigma,
    double expiry_days,
    double initial_balance,
    double fee_perc
) {
    std::vector<Trade> trades;
    std::vector<double> equity_curve;

    double balance = initial_balance;
    int position = 0;          // 0=flat, 1=long call, -1=long put
    double entry_price = 0.0;

    for (size_t i = 0; i < spot_prices.size(); i++) {
        double spot = spot_prices[i];
        double opt_price_call = option_price(spot, strike, sigma, expiry_days, true);
        double opt_price_put  = option_price(spot, strike, sigma, expiry_days, false);

        int signal = signals[i];
        if (signal == 1 && position == 0) {
            // Open long call
            position = 1;
            entry_price = opt_price_call;
            trades.push_back({"CALL", "BUY", entry_price, 0.0, 0.0, strike, true, expiry_days});
        } else if (signal == -1 && position == 0) {
            // Open long put
            position = -1;
            entry_price = opt_price_put;
            trades.push_back({"PUT", "BUY", entry_price, 0.0, 0.0, strike, false, expiry_days});
        } else if (signal == -1 && position == 1) {
            // Close call
            double pnl = (opt_price_put - entry_price) - (opt_price_put * fee_perc);
            balance += pnl;
            trades.back().exit_price = opt_price_put;
            trades.back().pnl = pnl;
            position = 0;
        } else if (signal == 1 && position == -1) {
            // Close put
            double pnl = (opt_price_call - entry_price) - (opt_price_call * fee_perc);
            balance += pnl;
            trades.back().exit_price = opt_price_call;
            trades.back().pnl = pnl;
            position = 0;
        }
        equity_curve.push_back(balance);
        expiry_days = std::max(0.0, expiry_days - 1.0 / 252.0); // decay daily
    }

    std::unordered_map<std::string, std::vector<double>> result;
    std::vector<double> pnls;
    for (auto& t : trades) {
        pnls.push_back(t.pnl);
    }
    result["equity_curve"] = equity_curve;
    result["pnl"] = pnls;
    return result;
}

// -----------------------------
// PyBind11 Binding
// -----------------------------
PYBIND11_MODULE(backtester_cpp, m) {
    m.doc() = "C++ Backtester for Options Trading (PyBind11)";
    m.def("backtest_options", &backtest_options, "Run an options backtest",
          py::arg("spot_prices"),
          py::arg("signals"),
          py::arg("strike"),
          py::arg("sigma") = 0.2,
          py::arg("expiry_days") = 30.0,
          py::arg("initial_balance") = 100000.0,
          py::arg("fee_perc") = 0.001);
}
