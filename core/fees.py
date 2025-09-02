def get_fees(broker: str, trade_price: float, qty: int) -> float:
    """
    Return transaction fees based on broker model
    trade_price: execution price
    qty: number of units
    """
    if broker.lower() == "zerodha":
        # Flat ₹20/order
        return 20.0
    elif broker.lower() == "ibkr":
        # 0.1% of trade value (min $1, assume $1 ~ ₹80)
        fees = 0.001 * trade_price * qty
        return max(fees, 80.0)  # ₹80 = $1
    elif broker.lower() == "angelone":
        # Flat ₹20/order for options, ₹0 for equities (simplified)
        return 20.0
    else:
        # Default flat ₹20
        return 20.0
