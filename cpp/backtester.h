#ifndef BACKTESTER_H
#define BACKTESTER_H

#include <vector>
#include <unordered_map>

// Run options backtest
std::unordered_map<std::string, std::vector<double>> backtest_options(
    const std::vector<double>& spot_prices,
    const std::vector<int>& signals,
    double strike,
    double sigma,
    double expiry_days,
    double initial_balance,
    double fee_perc
);

#endif // BACKTESTER_H
