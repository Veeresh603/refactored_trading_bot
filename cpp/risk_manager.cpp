#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Standard normal PDF
inline double norm_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

// Standard normal CDF (using error function erf)
inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

class RiskManager {
public:
    double max_risk_per_trade;
    double max_drawdown;
    double equity_high;

    RiskManager(double risk_per_trade = 0.01, double max_dd = 0.2)
        : max_risk_per_trade(risk_per_trade), max_drawdown(max_dd), equity_high(-1) {}

    // Position sizing
    int calculate_position_size(double balance, double stop_loss_points, double value_per_point=1.0) {
        if (stop_loss_points <= 0) return 0;
        double risk_amount = balance * max_risk_per_trade;
        int position_size = (int)(risk_amount / (stop_loss_points * value_per_point));
        return position_size > 0 ? position_size : 1;
    }

    // Stop Loss / Take Profit check
    bool check_stop_out(double equity) {
        if (equity_high < 0) equity_high = equity;
        if (equity > equity_high) equity_high = equity;
        return equity < equity_high * (1 - max_drawdown);
    }

    // Example probability calculation using CDF
    double probability_of_loss(double z_score) {
        return norm_cdf(z_score);  // instead of boost::math::cdf
    }

    // Example risk-reward metric using PDF
    double risk_reward_ratio(double z_score) {
        double reward = 1.0 - norm_cdf(z_score);
        double risk = norm_pdf(z_score);
        return reward / (risk + 1e-9);
    }
};

PYBIND11_MODULE(risk_manager, m) {
    py::class_<RiskManager>(m, "RiskManager")
        .def(py::init<double, double>(), py::arg("risk_per_trade") = 0.01, py::arg("max_dd") = 0.2)
        .def("calculate_position_size", &RiskManager::calculate_position_size)
        .def("check_stop_out", &RiskManager::check_stop_out)
        .def("probability_of_loss", &RiskManager::probability_of_loss)
        .def("risk_reward_ratio", &RiskManager::risk_reward_ratio);
}
