"""
Offline training with historical market data
"""

import pandas as pd
from ai.feature_engineer import add_indicators


def prepare_offline_data(filepath="data/nifty50.csv"):
    df = pd.read_csv(filepath)
    df = add_indicators(df)
    print("📊 Offline dataset prepared:", df.head())
    return df
# Offline training script placeholder
