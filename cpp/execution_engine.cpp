#include "execution_engine.h"
#include <iostream>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// --------------------------
// ExecutionEngine Methods
// --------------------------
ExecutionEngine::ExecutionEngine(double initial_balance) {
    balance = initial_balance;
    margin_used = 0.0;
}

// --------------------------
// Place Order
// --------------------------
void ExecutionEngine::place_order(const std::string& symbol, int qty, double price,
                                  double strike, double sigma, bool is_call, double expiry_days) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    if (qty <= 0) throw std::invalid_argument("Quantity must be > 0");
    if (price <= 0) throw std::invalid_argument("Price must be > 0");

    double cost = price * qty;
    if (balance < cost) {
        throw std::runtime_error("Insufficient balance for order");
    }

    auto it = positions.find(symbol);
    if (it == positions.end()) {
        Position pos = {symbol, qty, price, strike, sigma, is_call, expiry_days};
        positions[symbol] = pos;
    } else {
        Position& pos = it->second;
        double new_total_cost = pos.avg_price * pos.qty + price * qty;
        pos.qty += qty;
        pos.avg_price = new_total_cost / pos.qty;
        pos.strike = strike;
        pos.sigma = sigma;
        pos.is_call = is_call;
        pos.expiry_days = expiry_days;
    }

    balance -= cost;
    margin_used += cost * 0.1;  // assume 10% margin requirement

    log_trade("BUY " + symbol + " x" + std::to_string(qty) + " @" + std::to_string(price));
}

// --------------------------
// Portfolio Greeks (Black-Scholes approximation)
// --------------------------
std::tuple<double,double,double,double> ExecutionEngine::portfolio_greeks(double spot) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    double delta=0, gamma=0, vega=0, theta=0;

    for (auto& kv : positions) {
        Position& pos = kv.second;
        if (pos.qty == 0) continue;

        double d1 = (std::log(spot / pos.strike) + (0.5 * pos.sigma * pos.sigma) * pos.expiry_days)
                    / (pos.sigma * std::sqrt(pos.expiry_days));
        double d2 = d1 - pos.sigma * std::sqrt(pos.expiry_days);

        double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double nd1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

        if (pos.is_call) {
            delta += Nd1 * pos.qty;
            gamma += (nd1 / (spot * pos.sigma * std::sqrt(pos.expiry_days))) * pos.qty;
            vega  += spot * nd1 * std::sqrt(pos.expiry_days) * pos.qty;
            theta += -(spot * nd1 * pos.sigma / (2 * std::sqrt(pos.expiry_days))) * pos.qty;
        } else {
            delta += (Nd1 - 1.0) * pos.qty;
            gamma += (nd1 / (spot * pos.sigma * std::sqrt(pos.expiry_days))) * pos.qty;
            vega  += spot * nd1 * std::sqrt(pos.expiry_days) * pos.qty;
            theta += -(spot * nd1 * pos.sigma / (2 * std::sqrt(pos.expiry_days))) * pos.qty;
        }
    }
    return {delta, gamma, vega, theta};
}

// --------------------------
// Account Status
// --------------------------
std::unordered_map<std::string,double> ExecutionEngine::account_status(double spot) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    double unrealized = 0.0;
    for (auto& kv : positions) {
        Position& pos = kv.second;
        double current_price = std::max(spot - pos.strike, 0.0);
        if (!pos.is_call) current_price = std::max(pos.strike - spot, 0.0);
        unrealized += (current_price - pos.avg_price) * pos.qty;
    }

    std::unordered_map<std::string,double> status;
    status["balance"] = balance;
    status["margin_used"] = margin_used;
    status["realized"] = 0;  // placeholder
    status["unrealized"] = unrealized;
    status["total"] = unrealized;
    return status;
}

// --------------------------
// Trade Logging
// --------------------------
void ExecutionEngine::log_trade(const std::string& entry) {
    trade_log.push_back(entry);
}

std::vector<std::string> ExecutionEngine::get_trade_log() {
    return trade_log;
}

// --------------------------
// Reset Engine
// --------------------------
void ExecutionEngine::reset(double new_balance) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    positions.clear();
    balance = new_balance;
    margin_used = 0;
    trade_log.clear();
}

// --------------------------
// Global Engine Instance
// --------------------------
ExecutionEngine engine(100000.0);
