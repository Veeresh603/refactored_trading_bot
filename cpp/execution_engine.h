#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

struct Position {
    std::string symbol;
    int         qty;          // contracts (signed)
    double      strike;
    double      sigma;
    bool        is_call;
    int         expiry_days;  // >= 1

    // PnL bookkeeping
    double entry_price;       // premium at entry (per contract)
    double realized_pnl;      // accumulated on reductions
    double unrealized_pnl;    // refreshed on query
};

class ExecutionEngine {
private:
    double balance;
    double initial_balance;
    std::vector<Position> positions;

public:
    ExecutionEngine();
    void reset(double init_balance);

    void place_order(const std::string& symbol, int qty, double price,
                     double strike, double sigma, bool is_call, double expiry_days);

    std::unordered_map<std::string, double> account_status(double spot);

    // Greeks returned as per-contract totals (since qty is in contracts)
    std::tuple<double,double,double,double> portfolio_greeks(double spot);

    const std::vector<Position>& get_positions() const;

    // Compute and return positions with refreshed unrealized PnL (premium-based)
    std::vector<Position> get_positions_with_pnl(double spot);
    std::vector<std::string> get_trade_log();
};

// Helpers (exposed to cpp only; not required in Python)
double black_scholes_price(double spot, double strike, double T, double sigma, bool is_call);

#endif
