# backtesting/orderbook_loader.py
"""
Utilities to load orderbook-level snapshots from CSV files.

Expected CSV layout (one row per time-bar):
- time, close, volume, book
- where `book` is a JSON string containing keys "bids" and "asks".
  Example cell for book:
    '{"bids": [[19990.0, 10.0], [19980.0, 20.0]], "asks": [[20010.0, 8.0], [20020.0, 30.0]], "volume": 1000}'
"""

from __future__ import annotations
import json
import ast
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _parse_book_cell(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Try to parse a single CSV cell that should contain JSON-like book data.
    Accepts already-dict, JSON string, or Python literal string.
    Returns a dict with at least keys 'bids' and 'asks' or None on failure.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
        # Try JSON first
        try:
            obj = json.loads(raw)
            return obj
        except Exception:
            # Try python literal (single quotes)
            try:
                obj = ast.literal_eval(raw)
                return obj
            except Exception:
                logger.exception("Failed to parse book cell: %r", raw)
                return None
    # otherwise not supported
    return None


def load_orderbook_csv(path: str, book_col: str = "book") -> List[Dict[str, Any]]:
    """
    Load CSV at `path` and extract `levels_by_bar` for OrderBookSampler.
    Returns list indexed by row order. Each element is a dict {'bids': [(p,s),...], 'asks': [(p,s)...], 'volume': float}
    """
    df = pd.read_csv(path)
    if book_col not in df.columns:
        raise KeyError(f"Expected a column named '{book_col}' in the CSV. Found columns: {list(df.columns)}")

    levels_by_bar = []
    for idx, raw in enumerate(df[book_col].tolist()):
        parsed = _parse_book_cell(raw)
        if parsed is None:
            # If missing or unparsable, create an empty book
            parsed = {"bids": [], "asks": [], "volume": float(df.iloc[idx].get("volume", 0.0))}
        # Normalize structure: ensure bids/asks lists of (price, size) tuples and a numeric volume
        bids = parsed.get("bids", []) or []
        asks = parsed.get("asks", []) or []
        # Convert to list of tuples
        bids_t = [(float(p), float(s)) for (p, s) in bids]
        asks_t = [(float(p), float(s)) for (p, s) in asks]
        volume = float(parsed.get("volume", df.iloc[idx].get("volume", 0.0) or 0.0))
        levels_by_bar.append({"bids": bids_t, "asks": asks_t, "volume": volume})
    logger.info("Loaded %d orderbook bars from %s", len(levels_by_bar), path)
    return levels_by_bar
