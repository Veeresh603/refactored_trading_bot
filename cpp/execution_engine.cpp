#include "execution_engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Normal PDF/CDF ----------
static inline double norm_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}
static inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// ---------- Black–Scholes ----------
double black_scholes_price(double spot, double strike, double T, double sigma, bool is_call) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double vol_sqrt_T = sigma * std::sqrt(T);
    double d1 = (std::log(spot/strike) + 0.5*sigma*sigma*T) / vol_sqrt_T;
    double d2 = d1 - vol_sqrt_T;
    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    if (is_call) {
        return spot * Nd1 - strike * Nd2;
    } else {
        return strike * norm_cdf(-d2) - spot * norm_cdf(-d1);
    }
}

static inline std::tuple<double,double,double,double> black_scholes_greeks(
    double spot, double strike, double T, double sigma, bool is_call
) {
    if (T <= 0.0 || sigma <= 0.0) return {0.0,0.0,0.0,0.0};
    double vol_sqrt_T = sigma * std::sqrt(T);
    double d1 = (std::log(spot/strike) + 0.5*sigma*sigma*T) / vol_sqrt_T;
    double d2 = d1 - vol_sqrt_T;

    double delta = is_call ? norm_cdf(d1) : (norm_cdf(d1) - 1.0);
    double gamma = norm_pdf(d1) / (spot * vol_sqrt_T);
    double vega  = spot * norm_pdf(d1) * std::sqrt(T);
    double theta = -(spot * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T));
    // (no carry/discount terms to keep it simple)
    return {delta, gamma, vega, theta};
}

// ==============================
// ExecutionEngine Implementation
// ==============================

ExecutionEngine::ExecutionEngine()
: balance(100000.0), initial_balance(100000.0) {}

void ExecutionEngine::reset(double init_balance) {
    initial_balance = init_balance;
    balance = init_balance;
    positions.clear();
}

void ExecutionEngine::place_order(const std::string& symbol, int qty, double price,
                                  double strike, double sigma, bool is_call, double expiry_days) {
    // qty is in CONTRACTS (signed). price is OPTION PREMIUM per contract.
    int safe_expiry = static_cast<int>(std::max(1.0, expiry_days));

    int remaining = qty; // signed
    // FIFO-lite reduction for existing matching legs
    for (auto &pos : positions) {
        if (remaining == 0) break;
        if (pos.symbol != symbol || pos.is_call != is_call || pos.strike != strike) continue;

        // reduce if opposite sign
        if ((pos.qty > 0 && remaining < 0) || (pos.qty < 0 && remaining > 0)) {
            int close_size = std::min(std::abs(pos.qty), std::abs(remaining));
            int sign_existing = (pos.qty > 0 ? 1 : -1);
            // realized per contract = (close_price - entry_price) * sign_of_existing
            double per_contract = (price - pos.entry_price) * sign_existing;
            pos.realized_pnl += per_contract * close_size;

            // shrink existing leg toward zero
            pos.qty -= sign_existing * close_size;
            // move remaining toward zero (add the opposite of existing sign)
            remaining += sign_existing * close_size;
        }
    }

    // If net add remains, append a new leg with entry price
    if (remaining != 0) {
        Position pos{symbol, remaining, strike, sigma, is_call, safe_expiry,
                     price, 0.0, 0.0};
        positions.push_back(pos);
    }

    std::cout << "[TRACE] Placed order: " << symbol
              << " qty=" << qty
              << " @" << price
              << " strike=" << strike
              << " sigma=" << sigma
              << " expiry_days=" << safe_expiry
              << std::endl;
}

std::unordered_map<std::string, double> ExecutionEngine::account_status(double spot) {
    double realized = 0.0;
    double unrealized = 0.0;

    for (auto &pos : positions) {
        double T = std::max(1, pos.expiry_days) / 365.0;
        double cur = black_scholes_price(spot, pos.strike, T, pos.sigma, pos.is_call);
        pos.unrealized_pnl = (cur - pos.entry_price) * pos.qty; // signed qty
        realized   += pos.realized_pnl;
        unrealized += pos.unrealized_pnl;
    }

    return {
        {"balance", balance},
        {"total", balance + realized + unrealized},
        {"realized", realized},
        {"unrealized", unrealized}
    };
}

std::tuple<double,double,double,double> ExecutionEngine::portfolio_greeks(double spot) {
    double total_delta=0.0, total_gamma=0.0, total_vega=0.0, total_theta=0.0;

    for (auto &pos : positions) {
        double T = std::max(1, pos.expiry_days) / 365.0;
        auto [d,g,v,t] = black_scholes_greeks(spot, pos.strike, T, pos.sigma, pos.is_call);
        total_delta += d * pos.qty;   // qty is contracts
        total_gamma += g * pos.qty;
        total_vega  += v * pos.qty;
        total_theta += t * pos.qty;
    }
    return {total_delta,total_gamma,total_vega,total_theta};
}

const std::vector<Position>& ExecutionEngine::get_positions() const {
    return positions;
}

std::vector<Position> ExecutionEngine::get_positions_with_pnl(double spot) {
    std::vector<Position> out;
    out.reserve(positions.size());
    for (auto &pos : positions) {
        Position p = pos;
        double T = std::max(1, p.expiry_days) / 365.0;
        double cur = black_scholes_price(spot, p.strike, T, p.sigma, p.is_call);
        p.unrealized_pnl = (cur - p.entry_price) * p.qty;
        out.push_back(p);
    }
    return out;
}
