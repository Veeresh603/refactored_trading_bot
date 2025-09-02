#pragma once
#include <string>
#include <vector>

// Run a backtest on historical prices with a given strategy
double backtest_strategy(const std::vector<double> &prices, const std::string &strategy);
