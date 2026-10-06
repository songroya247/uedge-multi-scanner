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
BAND_MULT = 1.5    # standard deviations for channel width
INTERVAL = "1h"
PERIOD = "15d"     # 5d wasn't enough for indices -- they only trade market hours,
                   # so 5 calendar days was coming up short of 40 hourly bars

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
    body_r, lower_r, upper_r = body / rng, lower_wick / rng, upper_wick / rng

    # Ratios of the whole candle range, not just a multiple of the body --
    # avoids a near-zero body trivially passing the old body*2 check.
    if body_r < 0.3 and lower_r > 0.6 and upper_r < 0.15:
        return "buy", "Hammer"
    if body_r < 0.3 and upper_r > 0.6 and lower_r < 0.15:
        return "sell", "Shooting star"
    if len(c) >= 2:
        if c[-2] > o[-2] and c[-1] < o[-1] and o[-1] > c[-2] and c[-1] < o[-2]:
            return "sell", "Bearish engulfing"
        if c[-2] < o[-2] and c[-1] > o[-1] and o[-1] < c[-2] and c[-1] > o[-2]:
            return "buy", "Bullish engulfing"
    return None, None


STATE_FILE = "state.json"
RESULTS_FILE = "results.json"
MAX_RESULTS = 50  # how many recent alerts the Mini App feed keeps


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default
    return default


def scan():
    state = load_json(STATE_FILE, {})
    results = load_json(RESULTS_FILE, [])
    new_count, skipped = 0, 0

    for sym in SYMBOLS:
        try:
            df = fetch(sym)
            if df is None or len(df) < LOOKBACK:
                skipped += 1
                print(f"skipped {sym}: only {0 if df is None else len(df)} bars (need {LOOKBACK})")
                continue
            closes = df["Close"].values[-LOOKBACK:]
            price = float(closes[-1])
            name = pretty(sym)
            cls = asset_class(sym)
            candle_time = df.index[-1].isoformat()  # identifies the current (maybe still-forming) candle

            rc_dir = regression_signal(closes)
            if rc_dir:
                key = f"{sym}:regression"
                if state.get(key) != candle_time:  # only alert once per candle, not once per run
                    state[key] = candle_time
                    new_count += 1
                    alert = {"pair": name, "direction": rc_dir, "strategy": "regression",
                              "asset": cls, "price": round(price, 5),
                              "time": datetime.now(timezone.utc).isoformat()}
                    results.insert(0, alert)
                    send_telegram(f"{rc_dir.upper()} {name} - regression channel @ {alert['price']}")

            cs_dir, pattern = candlestick_signal(df)
            if cs_dir:
                key = f"{sym}:candlestick"
                if state.get(key) != candle_time:
                    state[key] = candle_time
                    new_count += 1
                    alert = {"pair": name, "direction": cs_dir, "strategy": "candlestick",
                              "pattern": pattern, "asset": cls, "price": round(price, 5),
                              "time": datetime.now(timezone.utc).isoformat()}
                    results.insert(0, alert)
                    send_telegram(f"{cs_dir.upper()} {name} - {pattern} @ {alert['price']}")
        except Exception as e:
            print(f"skipped {sym}: {e}")

    results = results[:MAX_RESULTS]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"scan complete: {new_count} new alert(s), {skipped} symbol(s) skipped, "
          f"{len(results)} total kept in the feed")


if __name__ == "__main__":
    scan()
