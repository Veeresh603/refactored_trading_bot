class DataManager:
    def __init__(self, kite):
        self.kite = kite
        self.data = {}

    def start_streaming(self, api_key, access_token, tokens, on_tick_callback):
        """
        tokens: list of instrument tokens to stream
        """
        from kiteconnect import KiteTicker
        kws = KiteTicker(api_key, access_token)

        def on_ticks(ws, ticks):
            # Group ticks by instrument_token
            for token in set([t["instrument_token"] for t in ticks]):
                token_ticks = [t for t in ticks if t["instrument_token"] == token]
                self.data[token] = token_ticks
            on_tick_callback(ticks)

        def on_connect(ws, response):
            ws.subscribe(tokens)

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.connect(threaded=True)
