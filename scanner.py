import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ---- Symbols ----
FOREX = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
         "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"]
INDICES = ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
SYMBOLS = FOREX + INDICES + CRYPTO

INDEX_NAMES = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "Nasdaq",
               "^FTSE": "FTSE 100", "^GDAXI": "DAX"}

# ---- Strategy settings ----
LOOKBACK = 40      # bars used to fit the regression channel
BAND_MULT = 2.0    # standard deviations for channel width
INTERVAL = "1h"
PERIOD = "5d"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def pretty(sym):
    if sym in INDEX_NAMES:
        return INDEX_NAMES[sym]
    if sym.endswith("=X"):
        base = sym[:-2]
        return base[:3] + "/" + base[3:]
    if sym.endswith("-USD"):
        return sym.replace("-USD", "/USD")
    return sym


def asset_class(sym):
    if sym in INDEX_NAMES:
        return "index"
    if sym.endswith("=X"):
        return "forex"
    return "crypto"


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"telegram send failed: {e}")


def fetch(sym):
    df = yf.download(sym, period=PERIOD, interval=INTERVAL, progress=False, timeout=10)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # yfinance adds a ticker level even for one symbol
    return df


def regression_signal(closes):
    x = np.arange(len(closes))
    slope, intercept = np.polyfit(x, closes, 1)
    line = slope * x + intercept
    std = (closes - line).std()
    upper, lower = line[-1] + BAND_MULT * std, line[-1] - BAND_MULT * std
    price = closes[-1]
    if price >= upper:
        return "sell"
    if price <= lower:
        return "buy"
    return None


def candlestick_signal(df):
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    body = abs(c[-1] - o[-1])
    rng = h[-1] - l[-1]
    if rng == 0:
        return None, None
    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])

    if lower_wick > body * 2 and upper_wick < body:
        return "buy", "Hammer"
    if upper_wick > body * 2 and lower_wick < body:
        return "sell", "Shooting star"
    if len(c) >= 2:
        if c[-2] > o[-2] and c[-1] < o[-1] and o[-1] > c[-2] and c[-1] < o[-2]:
            return "sell", "Bearish engulfing"
        if c[-2] < o[-2] and c[-1] > o[-1] and o[-1] < c[-2] and c[-1] > o[-2]:
            return "buy", "Bullish engulfing"
    return None, None


def scan():
    alerts = []
    for sym in SYMBOLS:
        try:
            df = fetch(sym)
            if df is None or len(df) < LOOKBACK:
                continue
            closes = df["Close"].values[-LOOKBACK:]
            price = float(closes[-1])
            name = pretty(sym)
            cls = asset_class(sym)

            rc_dir = regression_signal(closes)
            if rc_dir:
                alert = {"pair": name, "direction": rc_dir, "strategy": "regression",
                          "asset": cls, "price": round(price, 5),
                          "time": datetime.now(timezone.utc).isoformat()}
                alerts.append(alert)
                send_telegram(f"{rc_dir.upper()} {name} - regression channel @ {alert['price']}")

            cs_dir, pattern = candlestick_signal(df)
            if cs_dir:
                alert = {"pair": name, "direction": cs_dir, "strategy": "candlestick",
                          "pattern": pattern, "asset": cls, "price": round(price, 5),
                          "time": datetime.now(timezone.utc).isoformat()}
                alerts.append(alert)
                send_telegram(f"{cs_dir.upper()} {name} - {pattern} @ {alert['price']}")
        except Exception as e:
            print(f"skipped {sym}: {e}")

    with open("results.json", "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"scan complete: {len(alerts)} alert(s) across {len(SYMBOLS)} symbols")


if __name__ == "__main__":
    scan()
