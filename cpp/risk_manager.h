#pragma once
#include <string>

// Delta hedging (returns hedge size)
int hedge_delta(double total_delta, int lot_size);

// Gamma hedging
void hedge_gamma(const std::string &asset, double atm, double step);

// Vega hedging
void hedge_vega(const std::string &asset, double atm);
