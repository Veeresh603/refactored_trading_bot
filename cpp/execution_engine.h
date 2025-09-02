#pragma once
#include <string>
#include <map>

// Place an order
bool place_order(const std::string &symbol, int qty, double price,
                 const std::string &side, double strike, double sigma,
                 bool is_call, double expiry_days);

// Portfolio Greeks (delta, gamma, vega, theta)
std::tuple<double, double, double, double> portfolio_greeks(double spot);

// Unrealized and total PnL
double unrealized_pnl(double spot);
double total_pnl(double spot);

// Account status (balance, margin_used, realized, unrealized, total)
std::map<std::string, double> account_status(double spot);
