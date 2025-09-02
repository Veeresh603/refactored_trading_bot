#pragma once
#include <vector>

// Relative Strength Index
std::vector<double> rsi(const std::vector<double> &prices, int period);

// Simple Moving Average
std::vector<double> sma(const std::vector<double> &prices, int period);
