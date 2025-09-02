#include <cmath>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

    // Stop Loss / Take Profit
    double set_stop_loss(double entry_price, double stop_loss_points, std::string direction="BUY") {
        if (direction == "BUY") return entry_price - stop_loss_points;
        else return entry_price + stop_loss_points;
    }

    double set_take_profit(double entry_price, double take_profit_points, std::string direction="BUY") {
        if (direction == "BUY") return entry_price + take_profit_points;
        else return entry_price - take_profit_points;
    }

    // -------------------------------
    // Options Greeks
    // -------------------------------
    double calc_delta(bool is_call, double S, double K, double T, double r, double sigma);
    double calc_gamma(double S, double K, double T, double r, double sigma);
    double calc_vega(double S, double K, double T, double r, double sigma);
    double calc_theta(bool is_call, double S, double K, double T, double r, double sigma);

    // Portfolio aggregation
    double portfolio_delta(const std::vector<double>& deltas, const std::vector<int>& qty) {
        double total = 0;
        for (size_t i=0; i<deltas.size(); i++) total += deltas[i] * qty[i];
        return total;
    }

    // Hedge with futures
    int hedge_with_futures(double total_delta, double lot_size=50) {
        return (int) round(-total_delta / lot_size);
    }

    // Drawdown check
    bool check_drawdown(double current_equity) {
        if (equity_high < 0) equity_high = current_equity;
        double drawdown = (equity_high - current_equity) / equity_high;
        if (drawdown > max_drawdown) return false;
        if (current_equity > equity_high) equity_high = current_equity;
        return true;
    }
};

// -------------------------------
// Black-Scholes Greeks
// -------------------------------
#include <boost/math/distributions/normal.hpp>
using namespace boost::math;

double d1(double S, double K, double T, double r, double sigma) {
    return (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T));
}
double d2(double S, double K, double T, double r, double sigma) {
    return d1(S,K,T,r,sigma) - sigma*sqrt(T);
}

double RiskManager::calc_delta(bool is_call, double S, double K, double T, double r, double sigma) {
    normal_distribution<> N(0.0,1.0);
    if (is_call) return cdf(N, d1(S,K,T,r,sigma));
    else return cdf(N, d1(S,K,T,r,sigma)) - 1.0;
}

double RiskManager::calc_gamma(double S, double K, double T, double r, double sigma) {
    normal_distribution<> N(0.0,1.0);
    return pdf(N, d1(S,K,T,r,sigma)) / (S * sigma * sqrt(T));
}

double RiskManager::calc_vega(double S, double K, double T, double r, double sigma) {
    normal_distribution<> N(0.0,1.0);
    return S * pdf(N, d1(S,K,T,r,sigma)) * sqrt(T) / 100.0;
}

double RiskManager::calc_theta(bool is_call, double S, double K, double T, double r, double sigma) {
    normal_distribution<> N(0.0,1.0);
    double d_1 = d1(S,K,T,r,sigma);
    double d_2 = d2(S,K,T,r,sigma);

    double term1 = -(S * pdf(N, d_1) * sigma) / (2 * sqrt(T));
    if (is_call) {
        return (term1 - r*K*exp(-r*T)*cdf(N,d_2)) / 365.0;
    } else {
        return (term1 + r*K*exp(-r*T)*cdf(N,-d_2)) / 365.0;
    }
}

// -------------------------------
// PYBIND11 MODULE
// -------------------------------
PYBIND11_MODULE(fastrisk, m) {
    py::class_<RiskManager>(m, "RiskManager")
        .def(py::init<double,double>(), py::arg("risk_per_trade")=0.01, py::arg("max_drawdown")=0.2)
        .def("calculate_position_size", &RiskManager::calculate_position_size)
        .def("set_stop_loss", &RiskManager::set_stop_loss)
        .def("set_take_profit", &RiskManager::set_take_profit)
        .def("calc_delta", &RiskManager::calc_delta)
        .def("calc_gamma", &RiskManager::calc_gamma)
        .def("calc_vega", &RiskManager::calc_vega)
        .def("calc_theta", &RiskManager::calc_theta)
        .def("portfolio_delta", &RiskManager::portfolio_delta)
        .def("hedge_with_futures", &RiskManager::hedge_with_futures)
        .def("check_drawdown", &RiskManager::check_drawdown);
}
