# tests/test_orderbook_sampler.py
from backtesting.orderbook_sampler import OrderBookSampler

def test_simple_orderbook_execute_full():
    # single bar with small book
    levels = [{
        "bids": [(99.0, 5.0), (98.0, 100.0)],
        "asks": [(101.0, 3.0), (102.0, 100.0)],
        "volume": 200.0
    }]
    s = OrderBookSampler(levels)
    res = s.execute(0, side=1, requested_units=2.0)  # buy 2 -> should take two from asks @101
    assert res["executed"] == 2.0
    assert abs(res["vwap"] - 101.0) < 1e-9
    # request bigger than asks -> partial
    res2 = s.execute(0, side=1, requested_units=10.0)
    assert res2["executed"] <= 10.0
    assert res2["remaining"] >= 0.0
