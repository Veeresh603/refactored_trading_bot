#include <vector>
#include <numeric>
#include <cmath>
using namespace std;

// Simple Moving Average
vector<double> sma(const vector<double>& prices, int window) {
    vector<double> result(prices.size(), NAN);
    if (prices.size() < window) return result;

    double sum = 0.0;
    for (int i = 0; i < prices.size(); i++) {
        sum += prices[i];
        if (i >= window) sum -= prices[i - window];
        if (i >= window - 1) result[i] = sum / window;
    }
    return result;
}

// Relative Strength Index
vector<double> rsi(const vector<double>& prices, int period = 14) {
    vector<double> result(prices.size(), NAN);
    if (prices.size() < period) return result;

    double gain = 0, loss = 0;
    for (int i = 1; i <= period; i++) {
        double diff = prices[i] - prices[i-1];
        if (diff > 0) gain += diff; else loss -= diff;
    }
    double avg_gain = gain / period;
    double avg_loss = loss / period;

    result[period] = 100 - (100 / (1 + (avg_gain / avg_loss)));

    for (int i = period+1; i < prices.size(); i++) {
        double diff = prices[i] - prices[i-1];
        if (diff > 0) {
            avg_gain = (avg_gain * (period - 1) + diff) / period;
            avg_loss = (avg_loss * (period - 1)) / period;
        } else {
            avg_gain = (avg_gain * (period - 1)) / period;
            avg_loss = (avg_loss * (period - 1) - diff) / period;
        }
        result[i] = 100 - (100 / (1 + (avg_gain / avg_loss)));
    }
    return result;
}
