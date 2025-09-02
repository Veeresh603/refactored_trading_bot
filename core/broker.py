"""
Broker module
- Handles Zerodha (Kite) authentication & order placement
- Can be extended to other brokers in the future
"""

import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv


class Broker:
    def __init__(self, broker="zerodha", api_key=None, api_secret=None, access_token=None):
        load_dotenv()  # Load .env variables

        self.broker = broker
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET")
        self.access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")

        if self.broker == "zerodha":
            self.kite = KiteConnect(api_key=self.api_key)
        else:
            raise NotImplementedError(f"Broker {broker} is not supported yet.")

    def generate_login_url(self):
        """
        Generate login URL for manual authentication (only needed once to fetch request_token).
        """
        if self.broker == "zerodha":
            return self.kite.login_url()

    def set_access_token(self, request_token=None):
        """
        Exchange request_token for access_token (manual login).
        Stores the access_token for further sessions.
        """
        if not request_token:
            raise ValueError("Request token is required for generating access token.")

        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            print(f"✅ Access Token generated: {self.access_token}")
            print("👉 Save this token in your .env file as KITE_ACCESS_TOKEN")
            return self.access_token
        except Exception as e:
            print(f"❌ Failed to generate session: {e}")
            return None

    def connect(self):
        """
        Connect using existing access_token (after first-time login).
        """
        if not self.access_token:
            raise ValueError("Access token not found. Run set_access_token() first.")

        self.kite.set_access_token(self.access_token)
        print("🔗 Connected to Zerodha Kite API")
        return self.kite

    def place_order(self, symbol, qty, order_type="BUY", price=None, variety="regular"):
        """
        Place an order via broker
        """
        try:
            order_id = self.kite.place_order(
                variety=variety,
                exchange="NSE",
                tradingsymbol=symbol,
                transaction_type=order_type,
                quantity=qty,
                product="MIS",  # intraday
                order_type="LIMIT" if price else "MARKET",
                price=price if price else 0
            )
            print(f"✅ Order placed. ID: {order_id}")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            print(f"❌ Order failed: {e}")
            return {"status": "error", "message": str(e)}
