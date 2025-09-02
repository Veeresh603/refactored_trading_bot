"""
Strategy engine to manage multiple strategies (rule-based + AI)
"""
class StrategyEngine:
    def __init__(self):
        self.strategies = []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def run(self, market_data):
        for strat in self.strategies:
            strat.evaluate(market_data)