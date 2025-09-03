#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

// --------------------------
// Position Struct
// --------------------------
struct Position {
    std::string symbol;
    int qty;
    double avg_price;
    double strike;
    double sigma;       // volatility
    bool is_call;
    double expiry_days;
};

// --------------------------
// Execution Engine Class
// --------------------------
class ExecutionEngine {
private:
    std::unordered_map<std::string, Position> positions;
    double balance;
    double margin_used;
    std::vector<std::string> trade_log;

public:
    ExecutionEngine(double initial_balance = 100000.0);

    void place_order(const std::string& symbol, int qty, double price,
                     double strike, double sigma, bool is_call, double expiry_days);

    std::tuple<double,double,double,double> portfolio_greeks(double spot);

    std::unordered_map<std::string,double> account_status(double spot);

    void log_trade(const std::string& entry);

    std::vector<std::string> get_trade_log();

    void reset(double new_balance);
};

// --------------------------
// Global Instance
// --------------------------
extern ExecutionEngine engine;

#endif // EXECUTION_ENGINE_H
