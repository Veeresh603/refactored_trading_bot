#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cmath>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <tuple>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

struct Position {
    std::string symbol;
    int qty;
    double avg_price;
    double strike;
    double sigma; // volatility
    bool is_call;
    double expiry_days;
};

class ExecutionEngine {
private:
    std::unordered_map<std::string, Position> portfolio;
    std::mutex mtx;

    double account_balance;
    double margin_used;
    double realized_pnl;

    const double margin_per_lot = 50000.0; // Example: ₹50k per lot
    const std::string log_file = "trades.csv";

    // --- CSV Logging Helper ---
    void log_trade(const std::string &symbol, int qty, double price, const std::string &side) {
        std::ofstream file;
        file.open(log_file, std::ios::app);
        if (!file.is_open()) return;

        // Current timestamp
        std::time_t now = std::time(nullptr);
        char buf[80];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        file << buf << "," << symbol << "," << side << "," << qty << "," << price << "," << realized_pnl << "\n";
        file.close();
    }

    // Black-Scholes helpers
    double norm_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
    double d1(double S, double K, double T, double r, double sigma) {
        return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    }
    double d2(double d1, double sigma, double T) { return d1 - sigma * sqrt(T); }

public:
    ExecutionEngine(double starting_capital=1000000.0) {
        account_balance = starting_capital;
        margin_used = 0;
        realized_pnl = 0;

        // Initialize CSV with header if empty
        std::ofstream file(log_file, std::ios::app);
        if (file.tellp() == 0) {
            file << "Time,Symbol,Side,Qty,Price,RealizedPnL\n";
        }
        file.close();
    }

    // Margin check
    bool check_margin(int lots) {
        double required = lots * margin_per_lot;
        if (margin_used + required > account_balance) {
            std::cerr << "❌ Margin exceeded! Required=" << required
                      << " Available=" << (account_balance - margin_used) << std::endl;
            return false;
        }
        margin_used += required;
        return true;
    }

    // Place order
    bool place_order(const std::string &symbol, int qty, double price, const std::string &side,
                     double strike=0, double sigma=0.2, bool is_call=true, double expiry_days=30) {
        std::lock_guard<std::mutex> lock(mtx);

        int lots = qty / 50;
        if (!check_margin(lots)) return false;

        if (qty <= 0 || price <= 0) {
            std::cerr << "❌ Invalid order params" << std::endl;
            return false;
        }

        if (portfolio.find(symbol) == portfolio.end()) {
            portfolio[symbol] = {symbol, 0, 0.0, strike, sigma, is_call, expiry_days};
        }

        Position &pos = portfolio[symbol];
        if (side == "BUY") {
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / (pos.qty + qty);
            pos.qty += qty;
        } else if (side == "SELL") {
            int closing_qty = std::min(pos.qty, qty);
            if (closing_qty > 0) {
                double pnl = (price - pos.avg_price) * closing_qty * (pos.is_call ? 1 : -1);
                realized_pnl += pnl;
            }
            pos.qty -= qty;
            if (pos.qty <= 0) pos.avg_price = 0;
        }

        // Log trade to CSV
        log_trade(symbol, qty, price, side);

        std::cout << "✅ Executed " << side << " " << qty << " of " << symbol
                  << " @ " << price << " | RealizedPnL=" << realized_pnl
                  << " | MarginUsed=" << margin_used << std::endl;
        return true;
    }

    // Greeks for one position
    std::tuple<double, double, double, double> calc_greeks(const Position &pos, double S, double r=0.05) {
        double T = pos.expiry_days / 365.0;
        double K = pos.strike;
        double sigma = pos.sigma;

        double d1v = d1(S, K, T, r, sigma);
        double d2v = d2(d1v, sigma, T);

        double delta = pos.is_call ? norm_cdf(d1v) : norm_cdf(d1v) - 1;
        double gamma = exp(-0.5 * d1v * d1v) / (S * sigma * sqrt(2 * M_PI * T));
        double vega = S * exp(-0.5 * d1v * d1v) * sqrt(T) / sqrt(2 * M_PI);
        double theta = -(S * sigma * exp(-0.5 * d1v * d1v)) / (2 * sqrt(2 * M_PI * T));

        return {delta * pos.qty, gamma * pos.qty, vega * pos.qty, theta * pos.qty};
    }

    // Portfolio Greeks
    std::tuple<double, double, double, double> portfolio_greeks(double S, double r=0.05) {
        std::lock_guard<std::mutex> lock(mtx);
        double d_sum=0, g_sum=0, v_sum=0, t_sum=0;
        for (auto &kv : portfolio) {
            auto [d, g, v, t] = calc_greeks(kv.second, S, r);
            d_sum += d; g_sum += g; v_sum += v; t_sum += t;
        }
        return {d_sum, g_sum, v_sum, t_sum};
    }

    // Unrealized PnL
    double unrealized_pnl(double spot) {
        std::lock_guard<std::mutex> lock(mtx);
        double unrealized = 0;
        for (auto &kv : portfolio) {
            Position &pos = kv.second;
            double mtm_price = spot; // Simplification: mark-to-market = spot
            unrealized += (mtm_price - pos.avg_price) * pos.qty * (pos.is_call ? 1 : -1);
        }
        return unrealized;
    }

    double total_pnl(double spot) {
        return realized_pnl + unrealized_pnl(spot);
    }

    void account_status(double spot, double* balance, double* used_margin,
                        double* realized, double* unrealized, double* total) {
        *balance = account_balance;
        *used_margin = margin_used;
        *realized = realized_pnl;
        *unrealized = unrealized_pnl(spot);
        *total = total_pnl(spot);
    }

    // Hedging
    int hedge_delta(double total_delta, int lot_size=50) {
        int lots = (int) round(total_delta / lot_size);
        if (lots != 0) {
            if (!check_margin(abs(lots))) return 0;
            std::string fut_symbol = "FUT_CONTRACT";
            std::string side = (lots > 0) ? "SELL" : "BUY";
            place_order(fut_symbol, abs(lots) * lot_size, 1.0, side);
            std::cout << "🛡️ Delta hedge executed: " << lots << " lots futures" << std::endl;
        }
        return lots;
    }

    void hedge_gamma(const std::string &asset, double atm, double step=50) {
        if (!check_margin(2)) return;
        place_order(asset + std::to_string((int)(atm-step)) + "CE", 50, 1.0, "BUY", atm-step, 0.2, true, 30);
        place_order(asset + std::to_string((int)atm) + "CE", 100, 1.0, "SELL", atm, 0.2, true, 30);
        place_order(asset + std::to_string((int)(atm+step)) + "CE", 50, 1.0, "BUY", atm+step, 0.2, true, 30);
        std::cout << "🦋 Gamma hedge executed (Butterfly)" << std::endl;
    }

    void hedge_vega(const std::string &asset, double atm) {
        if (!check_margin(2)) return;
        place_order(asset + std::to_string((int)atm) + "CE", 50, 1.0, "BUY", atm, 0.2, true, 30);
        place_order(asset + std::to_string((int)atm) + "PE", 50, 1.0, "BUY", atm, 0.2, false, 30);
        std::cout << "🎭 Vega hedge executed (Straddle)" << std::endl;
    }
};

// Global engine instance
ExecutionEngine engine(2000000.0); // start with ₹20L

// ----------------------------
// C API for Python wrapper
// ----------------------------
extern "C" {
    bool place_order(const char* symbol, int qty, double price, const char* side,
                     double strike, double sigma, bool is_call, double expiry_days) {
        return engine.place_order(symbol, qty, price, side, strike, sigma, is_call, expiry_days);
    }

    void portfolio_greeks(double S, double* delta, double* gamma, double* vega, double* theta) {
        auto [d, g, v, t] = engine.portfolio_greeks(S);
        *delta = d; *gamma = g; *vega = v; *theta = t;
    }

    int hedge_delta(double total_delta, int lot_size) {
        return engine.hedge_delta(total_delta, lot_size);
    }

    void hedge_gamma(const char* asset, double atm, double step) {
        engine.hedge_gamma(asset, atm, step);
    }

    void hedge_vega(const char* asset, double atm) {
        engine.hedge_vega(asset, atm);
    }

    double unrealized_pnl(double spot) {
        return engine.unrealized_pnl(spot);
    }

    double total_pnl(double spot) {
        return engine.total_pnl(spot);
    }

    void account_status(double spot, double* balance, double* used_margin,
                        double* realized, double* unrealized, double* total) {
        engine.account_status(spot, balance, used_margin, realized, unrealized, total);
    }
}
