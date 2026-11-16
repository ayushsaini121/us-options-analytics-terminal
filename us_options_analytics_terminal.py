"""
US OPTIONS ANALYTICS TERMINAL — ALPACA + YAHOO FINANCE
======================================================
Free-data configuration:
  • Alpaca: SPY equity quotes/bars and indicative US options snapshots/history.
  • Yahoo Finance: SPX index (^GSPC) and fallback quotes/chains/history.
  • Backtesting: synthetic Black–Scholes option marks on historical underlying bars.

Install:
    python -m pip install numpy pandas matplotlib requests yfinance

Set Alpaca credentials (PowerShell):
    $env:ALPACA_API_KEY="YOUR_KEY"
    $env:ALPACA_SECRET_KEY="YOUR_SECRET"

Set Alpaca credentials (Command Prompt):
    set ALPACA_API_KEY=YOUR_KEY
    set ALPACA_SECRET_KEY=YOUR_SECRET

Run:
    python us_options_analytics_terminal.py

IMPORTANT:
- Alpaca Basic options data is INDICATIVE, not consolidated OPRA BBO.
- Alpaca Basic equity real-time data is IEX-only, not consolidated SIP.
- Alpaca historical options data is available from February 2024; the
  indicative feed is a free derivative of OPRA (not actual OPRA quotes).
- SPX is an index, not an ETF. Alpaca's stock endpoints may not provide
  an SPX index quote/chain; Yahoo Finance is used as the SPX fallback.
- Backtests below use synthetic Black–Scholes marks, NOT historical option
  executions or quote prices. They are educational model simulations.
- This application does not place orders.
"""

import os
import sys
import math
import time
import re
import getpass
import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------- CONFIG -----------------------------

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "PKLVE5GW536LN2G4RGOKZGT35Y").strip()
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "9jZJc64o9va66tkokAgAfc5UquVRBwWZHwMPefdxnbru").strip()
ALPACA_DATA_URL = "https://data.alpaca.markets"
YAHOO_SYMBOLS = {"SPY": "SPY", "SPX": "^GSPC"}
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
CONTRACT_MULTIPLIER = 100


# ------------------------ BLACK-SCHOLES ---------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class OptionInputs:
    spot: float
    strike: float
    volatility: float
    rate: float
    years: float
    option_type: str = "call"
    dividend_yield: float = 0.0


class BlackScholes:
    """European option valuation with continuous dividend yield."""

    def __init__(self, inputs: OptionInputs):
        self.x = inputs
        self.x.option_type = str(self.x.option_type).lower()
        if self.x.spot <= 0 or self.x.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.x.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
        if self.x.years < 0:
            raise ValueError("Time to expiry cannot be negative.")
        if self.x.option_type not in ("call", "put"):
            raise ValueError("option_type must be call or put.")

    def _d1d2(self):
        x = self.x
        if x.years <= 0 or x.volatility <= 0:
            return None, None
        root_t = math.sqrt(x.years)
        d1 = (
            math.log(x.spot / x.strike)
            + (x.rate - x.dividend_yield + 0.5 * x.volatility**2) * x.years
        ) / (x.volatility * root_t)
        return d1, d1 - x.volatility * root_t

    def price(self):
        x = self.x
        if x.years <= 0:
            return max(x.spot - x.strike, 0.0) if x.option_type == "call" \
                else max(x.strike - x.spot, 0.0)
        if x.volatility == 0:
            fwd_pv = x.spot * math.exp(-x.dividend_yield * x.years)
            strike_pv = x.strike * math.exp(-x.rate * x.years)
            return max(fwd_pv - strike_pv, 0.0) if x.option_type == "call" \
                else max(strike_pv - fwd_pv, 0.0)
        d1, d2 = self._d1d2()
        if x.option_type == "call":
            return x.spot * math.exp(-x.dividend_yield*x.years)*norm_cdf(d1) \
                - x.strike * math.exp(-x.rate*x.years)*norm_cdf(d2)
        return x.strike * math.exp(-x.rate*x.years)*norm_cdf(-d2) \
            - x.spot * math.exp(-x.dividend_yield*x.years)*norm_cdf(-d1)

    def greeks(self):
        x = self.x
        d1, d2 = self._d1d2()
        if x.years <= 0 or x.volatility <= 0:
            itm = x.spot > x.strike if x.option_type == "call" else x.spot < x.strike
            delta = (1.0 if itm else 0.0) if x.option_type == "call" \
                else (-1.0 if itm else 0.0)
            return {"delta": delta, "gamma": 0.0, "theta": 0.0,
                    "vega": 0.0, "rho": 0.0}
        q, r, t, sig = x.dividend_yield, x.rate, x.years, x.volatility
        dq, dr = math.exp(-q*t), math.exp(-r*t)
        gamma = dq * norm_pdf(d1) / (x.spot * sig * math.sqrt(t))
        vega = x.spot * dq * norm_pdf(d1) * math.sqrt(t) / 100.0
        if x.option_type == "call":
            delta = dq * norm_cdf(d1)
            theta = (-x.spot*dq*norm_pdf(d1)*sig/(2*math.sqrt(t))
                     - r*x.strike*dr*norm_cdf(d2)
                     + q*x.spot*dq*norm_cdf(d1)) / 365.0
            rho = x.strike*t*dr*norm_cdf(d2)/100.0
        else:
            delta = dq * (norm_cdf(d1)-1)
            theta = (-x.spot*dq*norm_pdf(d1)*sig/(2*math.sqrt(t))
                     + r*x.strike*dr*norm_cdf(-d2)
                     - q*x.spot*dq*norm_cdf(-d1)) / 365.0
            rho = -x.strike*t*dr*norm_cdf(-d2)/100.0
        return {"delta": delta, "gamma": gamma, "theta": theta,
                "vega": vega, "rho": rho}

    def probability_itm_risk_neutral(self):
        x = self.x
        if x.years <= 0:
            return float(x.spot > x.strike) if x.option_type == "call" \
                else float(x.spot < x.strike)
        if x.volatility <= 0:
            fwd = x.spot * math.exp((x.rate-x.dividend_yield)*x.years)
            return float(fwd > x.strike) if x.option_type == "call" \
                else float(fwd < x.strike)
        _, d2 = self._d1d2()
        return norm_cdf(d2) if x.option_type == "call" else norm_cdf(-d2)


# Original class names retained for compatibility.
class EuropeanCall(BlackScholes):
    def __init__(self, S, K, sigma, r, T, drift=0.0):
        super().__init__(OptionInputs(S, K, sigma, r, T, "call"))


class EuropeanPut(BlackScholes):
    def __init__(self, S, K, sigma, r, T, drift=0.0):
        super().__init__(OptionInputs(S, K, sigma, r, T, "put"))


# ----------------------------- ALPACA -----------------------------

class AlpacaDataClient:
    """Small REST client using Alpaca's free indicative/IEX market-data feeds."""

    def __init__(self, key=None, secret=None):
        self.key = key or ALPACA_KEY
        self.secret = secret or ALPACA_SECRET
        self.session = requests.Session()
        if self.key and self.secret:
            self.session.headers.update({
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "accept": "application/json",
            })

    @property
    def configured(self):
        return bool(self.key and self.secret)

    def get(self, path, params=None):
        if not self.configured:
            raise RuntimeError(
                "Alpaca keys are missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )
        url = ALPACA_DATA_URL + path
        response = self.session.get(url, params=params or {}, timeout=30)
        if response.status_code >= 400:
            detail = response.text[:500]
            if response.status_code == 403:
                detail += " | Check your Alpaca market-data entitlement/feed."
            raise RuntimeError(f"Alpaca API {response.status_code}: {detail}")
        return response.json()

    def latest_stock_quotes(self, symbols=("SPY",), feed="iex"):
        return self.get("/v2/stocks/quotes/latest", {
            "symbols": ",".join(symbols), "feed": feed
        }).get("quotes", {})

    def latest_stock_bars(self, symbols=("SPY",), feed="iex"):
        return self.get("/v2/stocks/bars/latest", {
            "symbols": ",".join(symbols), "feed": feed
        }).get("bars", {})

    def stock_bars(self, symbol, start, end, timeframe="1Day", feed="iex"):
        params = {
            "symbols": symbol, "timeframe": timeframe,
            "start": start, "end": end, "feed": feed,
            "adjustment": "all", "limit": 10000, "sort": "asc",
        }
        rows, token = [], None
        while True:
            if token:
                params["page_token"] = token
            data = self.get("/v2/stocks/bars", params)
            rows.extend(data.get("bars", {}).get(symbol, []))
            token = data.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(None)
        return frame.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close",
                                     "v":"Volume"}).set_index("timestamp")

    def option_chain(self, underlying, feed="indicative", max_pages=20):
        params = {"feed": feed, "limit": 1000}
        snapshots, token, page = {}, None, 0
        while page < max_pages:
            if token:
                params["page_token"] = token
            data = self.get(f"/v1beta1/options/snapshots/{underlying}", params)
            snapshots.update(data.get("snapshots", {}))
            token = data.get("next_page_token")
            page += 1
            if not token:
                break
        return snapshots

    def historical_option_bars(self, occ_symbol, start, end,
                               timeframe="1Day", feed="indicative"):
        params = {
            "symbols": occ_symbol, "timeframe": timeframe,
            "start": start, "end": end, "feed": feed,
            "limit": 10000, "sort": "asc",
        }
        rows, token = [], None
        while True:
            if token:
                params["page_token"] = token
            data = self.get("/v1beta1/options/bars", params)
            rows.extend(data.get("bars", {}).get(occ_symbol, []))
            token = data.get("next_page_token")
            if not token:
                break
        return pd.DataFrame(rows)


# ----------------------------- YAHOO -------------------------------

def yahoo_ticker(symbol):
    return yf.Ticker(YAHOO_SYMBOLS.get(symbol.upper(), symbol))


def yahoo_quote(symbol):
    ticker = yahoo_ticker(symbol)
    info = {}
    try:
        info = ticker.fast_info
    except Exception:
        pass
    last = None
    for key in ("lastPrice", "last_price", "regularMarketPrice"):
        try:
            value = info.get(key)
            if value is not None and np.isfinite(float(value)):
                last = float(value)
                break
        except Exception:
            continue
    if last is None:
        hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty:
            last = float(hist["Close"].dropna().iloc[-1])
    return {"symbol": symbol.upper(), "last": last, "source": "Yahoo Finance"}


def yahoo_option_chain(symbol, expiration=None):
    # Yahoo uses ^SPX for the index option chain; ^GSPC is its cash-index history ticker.
    ticker = yf.Ticker("^SPX" if symbol.upper() == "SPX" else "SPY")
    expiries = list(ticker.options or [])
    if not expiries:
        raise RuntimeError(f"Yahoo Finance returned no option expirations for {symbol}.")
    chosen = expiration if expiration in expiries else expiries[0]
    chain = ticker.option_chain(chosen)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["optionType"], puts["optionType"] = "call", "put"
    result = pd.concat([calls, puts], ignore_index=True)
    result["expiration"] = chosen
    return result, expiries


def download_underlying_history(symbol, period="5y", alpaca=None):
    """Prefer Alpaca IEX daily bars for SPY; use Yahoo as robust fallback."""
    if symbol.upper() == "SPY" and alpaca and alpaca.configured:
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * 7)
        try:
            frame = alpaca.stock_bars("SPY", start.isoformat(), end.isoformat(),
                                      timeframe="1Day", feed="iex")
            if not frame.empty:
                frame = frame[["Open","High","Low","Close","Volume"]].copy()
                frame.index = pd.to_datetime(frame.index)
                return frame, "Alpaca IEX daily bars"
        except Exception as exc:
            print(f"Alpaca history unavailable; falling back to Yahoo: {exc}")
    frame = yahoo_ticker(symbol).history(period=period, interval="1d",
                                         auto_adjust=True)
    if frame.empty:
        raise RuntimeError(f"No historical underlying data returned for {symbol}.")
    return frame[["Open","High","Low","Close","Volume"]].dropna(), "Yahoo Finance"


# ------------------------- MARKET / CHAIN --------------------------

def market_monitor():
    print("\nLIVE MARKET MONITOR — SPY / SPX")
    print("Alpaca SPY feed: IEX (free tier; not consolidated SIP).")
    print("Yahoo SPX feed: ^GSPC index quote; may be delayed.")
    alpaca = AlpacaDataClient()
    interval = max(5, int(input("Refresh interval in seconds [10]: ") or "10"))
    duration = int(input("Run duration in seconds [60; 0=until Ctrl+C]: ") or "60")
    started = time.monotonic()
    try:
        while True:
            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n--- {stamp} local time ---")
            if alpaca.configured:
                try:
                    quotes = alpaca.latest_stock_quotes(("SPY",), feed="iex")
                    q = quotes.get("SPY", {})
                    if q:
                        print("SPY | Alpaca IEX | "
                              f"Bid {q.get('bp')} x {q.get('bs')} | "
                              f"Ask {q.get('ap')} x {q.get('as')} | "
                              f"Quote time {q.get('t')}")
                    else:
                        print("SPY | Alpaca IEX | No quote returned")
                except Exception as exc:
                    print(f"SPY | Alpaca error: {exc}")
            else:
                print("SPY | Alpaca unavailable — configure API keys for IEX quotes.")
            for symbol in ("SPY", "SPX"):
                try:
                    quote = yahoo_quote(symbol)
                    print(f"{symbol} | Yahoo Finance | Last: {quote['last']}")
                except Exception as exc:
                    print(f"{symbol} | Yahoo error: {exc}")
            if duration > 0 and time.monotonic() - started >= duration:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMarket monitor stopped by user.")
    print("\nReminder: Yahoo quotes may be delayed; Alpaca Basic equity data is IEX-only.")


def parse_occ_symbol(occ):
    """Parse standard OCC option symbol, e.g. SPY260918C00600000."""
    match = re.match(r"^([A-Z0-9.]+?)(\d{6})([CP])(\d{8})$", occ.upper())
    if not match:
        return {}
    root, yymmdd, cp, strike = match.groups()
    expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
    return {"root": root, "expiration": expiry.isoformat(),
            "type": "call" if cp == "C" else "put",
            "strike": int(strike) / 1000.0}


def alpaca_chain_frame(snapshots):
    rows = []
    for occ, snap in snapshots.items():
        parsed = parse_occ_symbol(occ)
        if not parsed:
            continue
        quote = snap.get("latestQuote") or {}
        trade = snap.get("latestTrade") or {}
        greeks = snap.get("greeks") or {}
        rows.append({
            "contractSymbol": occ, **parsed,
            "bid": quote.get("bp"), "ask": quote.get("ap"),
            "bidSize": quote.get("bs"), "askSize": quote.get("as"),
            "lastPrice": trade.get("p"),
            "impliedVolatility": snap.get("impliedVolatility"),
            "delta": greeks.get("delta"), "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"), "vega": greeks.get("vega"),
            "rho": greeks.get("rho"),
            "quoteTime": quote.get("t"), "tradeTime": trade.get("t"),
            "dataFeed": "Alpaca indicative",
        })
    return pd.DataFrame(rows)


def show_option_chain():
    symbol = input("Underlying [SPY or SPX] (SPY): ").strip().upper() or "SPY"
    if symbol not in ("SPY", "SPX"):
        raise ValueError("Choose SPY or SPX.")
    alpaca = AlpacaDataClient()
    frame = pd.DataFrame()
    if symbol == "SPY" and alpaca.configured:
        try:
            print("Requesting Alpaca indicative option snapshots...")
            frame = alpaca_chain_frame(alpaca.option_chain("SPY", feed="indicative"))
        except Exception as exc:
            print(f"Alpaca chain unavailable; trying Yahoo Finance: {exc}")
    if frame.empty:
        frame, expiries = yahoo_option_chain(symbol)
        frame["dataFeed"] = "Yahoo Finance"
        print("Yahoo expirations:", ", ".join(expiries[:15]))
    if frame.empty:
        print("No chain rows returned.")
        return
    if "expiration" in frame.columns:
        expiries = sorted(frame["expiration"].dropna().astype(str).unique())
        print("Available expirations:", ", ".join(expiries[:20]))
        selected = input(f"Filter expiration [{expiries[0]}] (blank=all): ").strip()
        if selected:
            frame = frame[frame["expiration"].astype(str) == selected]
    if "strike" in frame.columns:
        frame = frame.sort_values(["expiration", "strike", "type"] if "type" in frame.columns
                                  else ["strike"])
    print(f"\nReturned {len(frame)} contracts.")
    print(frame.head(60).to_string(index=False))
    out = f"{symbol.lower()}_option_chain.csv"
    frame.to_csv(out, index=False)
    print(f"\nFull chain saved to {out}")


def historical_indicative_option_data():
    print("\nALPACA HISTORICAL INDICATIVE OPTION BARS")
    print("Enter a standard OCC contract symbol, e.g. SPY260918C00600000")
    occ = input("OCC option symbol: ").strip().upper()
    if not parse_occ_symbol(occ):
        raise ValueError("That does not look like a standard OCC option symbol.")
    start = input("Start date YYYY-MM-DD [2025-01-01]: ").strip() or "2025-01-01"
    end = input("End date YYYY-MM-DD [today]: ").strip() or dt.date.today().isoformat()
    timeframe = input("Timeframe [1Day]: ").strip() or "1Day"
    alpaca = AlpacaDataClient()
    df = alpaca.historical_option_bars(occ, start, end, timeframe, "indicative")
    if df.empty:
        print("No historical indicative bars returned for that contract/date range.")
        print("Check the OCC symbol, date range, and Alpaca access. Options history starts Feb 2024.")
        return
    df["timestamp"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp")
    out = f"{occ.lower()}_alpaca_indicative_history.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {len(df)} rows to {out}")
    print("These are Alpaca indicative-feed bars, not consolidated OPRA records.")


# ---------------------- SYNTHETIC BACKTEST -------------------------

def historical_backtest():
    print("\nSYNTHETIC BLACK-SCHOLES OPTIONS BACKTEST")
    symbol = input("Underlying [SPY or SPX] (SPY): ").strip().upper() or "SPY"
    if symbol not in ("SPY", "SPX"):
        raise ValueError("Choose SPY or SPX.")
    side = input("Option side [call/put] (call): ").strip().lower() or "call"
    if side not in ("call", "put"):
        raise ValueError("Choose call or put.")
    fast = int(input("Fast SMA [20]: ") or "20")
    slow = int(input("Slow SMA [50]: ") or "50")
    dte = int(input("Option tenor in calendar days [30]: ") or "30")
    max_hold = int(input("Maximum holding days [10]: ") or "10")
    starting_cash = float(input("Starting capital USD [10000]: ") or "10000")
    contracts = int(input("Contracts per trade [1]: ") or "1")
    fee_per_contract = float(input("Fee per contract per side USD [0.65]: ") or "0.65")
    if fast < 2 or slow <= fast or dte < 2 or max_hold < 1 or contracts < 1:
        raise ValueError("Require slow > fast >= 2, DTE >= 2, max hold >= 1, contracts >= 1.")

    alpaca = AlpacaDataClient()
    bars, source = download_underlying_history(symbol, period="7y", alpaca=alpaca)
    bars = bars.copy()
    bars["SMA_fast"] = bars["Close"].rolling(fast).mean()
    bars["SMA_slow"] = bars["Close"].rolling(slow).mean()
    bars["Volatility"] = bars["Close"].pct_change().rolling(20).std() * math.sqrt(TRADING_DAYS)
    bars = bars.dropna().copy()
    if len(bars) < slow + max_hold + 5:
        raise RuntimeError("Not enough historical data for these SMA settings.")

    cash = starting_cash
    position = None
    trades = []
    dates = list(bars.index)

    def close_position(date, spot, reason):
        nonlocal cash, position
        if position is None:
            return
        days_held = max(0, (date - position["entry_date"]).days)
        remaining = max(0, position["dte"] - days_held)
        years = remaining / 365.0
        mark = BlackScholes(OptionInputs(
            spot, position["strike"], position["vol"], RISK_FREE_RATE,
            years, side, 0.0
        )).price()
        exit_value = mark * CONTRACT_MULTIPLIER * contracts
        exit_fee = fee_per_contract * contracts
        cash += exit_value - exit_fee
        pnl = (mark - position["entry_premium"]) * CONTRACT_MULTIPLIER * contracts \
              - 2 * fee_per_contract * contracts
        trades.append({
            "entry_date": position["entry_date"].date(),
            "exit_date": date.date(), "side": side,
            "entry_spot": position["entry_spot"], "exit_spot": spot,
            "strike": position["strike"], "entry_vol": position["vol"],
            "entry_premium": position["entry_premium"], "exit_model_mark": mark,
            "contracts": contracts, "days_held": days_held,
            "exit_reason": reason, "net_pnl_usd": pnl,
        })
        position = None

    for i in range(1, len(bars)):
        row = bars.iloc[i]
        prev = bars.iloc[i - 1]
        date = bars.index[i]
        spot = float(row["Close"])
        signal = float(prev["SMA_fast"]) > float(prev["SMA_slow"])
        reverse_signal = float(prev["SMA_fast"]) < float(prev["SMA_slow"])

        if position is not None:
            held = (date - position["entry_date"]).days
            if reverse_signal:
                close_position(date, spot, "SMA signal reversed")
            elif held >= max_hold:
                close_position(date, spot, "Maximum holding period")
        if position is None and signal:
            vol = float(prev["Volatility"])
            if not np.isfinite(vol) or vol <= 0:
                continue
            strike = spot * (1.0 if side == "call" else 1.0)
            premium = BlackScholes(OptionInputs(
                spot, strike, vol, RISK_FREE_RATE, dte/365.0, side, 0.0
            )).price()
            cost = premium * CONTRACT_MULTIPLIER * contracts \
                   + fee_per_contract * contracts
            if cost <= cash:
                cash -= cost
                position = {
                    "entry_date": date, "entry_spot": spot, "strike": strike,
                    "vol": vol, "dte": dte, "entry_premium": premium,
                }

    if position is not None:
        close_position(bars.index[-1], float(bars["Close"].iloc[-1]), "End of data")
    if not trades:
        print("No trades generated. Try a different SMA/side/period configuration.")
        return

    results = pd.DataFrame(trades)
    total_pnl = float(results["net_pnl_usd"].sum())
    wins = int((results["net_pnl_usd"] > 0).sum())
    losses = int((results["net_pnl_usd"] < 0).sum())
    gross_profit = float(results.loc[results.net_pnl_usd > 0, "net_pnl_usd"].sum())
    gross_loss = abs(float(results.loc[results.net_pnl_usd < 0, "net_pnl_usd"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    final_equity = starting_cash + total_pnl
    print("\nBACKTEST SUMMARY")
    print(f"Underlying:          {symbol}")
    print(f"Historical source:   {source}")
    print(f"Bars tested:         {len(bars)}")
    print(f"Trades:              {len(results)}")
    print(f"Win rate:            {wins/len(results)*100:.2f}%")
    print(f"Net P&L:             ${total_pnl:,.2f}")
    print(f"Starting capital:    ${starting_cash:,.2f}")
    print(f"Ending equity:       ${final_equity:,.2f}")
    print(f"Return:              {(final_equity/starting_cash-1)*100:.2f}%")
    print(f"Profit factor:       {profit_factor:.2f}")
    print(f"Winning / losing:    {wins} / {losses}")
    print(f"Option model:        Black–Scholes synthetic marks")
    print(f"Risk-free rate:      {RISK_FREE_RATE:.2%} (fixed assumption)")
    print("WARNING: Not a historical option-quote backtest; no OPRA fills,")
    print("        bid/ask spread, slippage, early exercise or assignment model.")
    out = f"{symbol.lower()}_synthetic_bs_backtest.csv"
    results.to_csv(out, index=False)
    print(f"\nTrade log saved to {out}")
    equity = starting_cash + results["net_pnl_usd"].cumsum()
    plt.figure(figsize=(11, 5))
    plt.plot(pd.to_datetime(results["exit_date"]), equity, label="Modelled equity")
    plt.axhline(starting_cash, linestyle="--", label="Starting capital")
    plt.title(f"{symbol} Synthetic Black–Scholes Backtest")
    plt.xlabel("Exit date")
    plt.ylabel("Equity (USD)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------- PRICER / PAYOFF ----------------------------

def interactive_pricer():
    print("\nBLACK-SCHOLES PRICER + FIVE GREEKS")
    spot = float(input("Underlying spot [600]: ") or "600")
    strike = float(input("Strike [600]: ") or "600")
    vol = float(input("Annual volatility as % [20]: ") or "20") / 100
    rate = float(input("Risk-free rate as % [4]: ") or "4") / 100
    dte = float(input("Days to expiry [30]: ") or "30")
    dividend = float(input("Annual dividend yield as % [0]: ") or "0") / 100
    kind = input("Call or put [call]: ").strip().lower() or "call"
    model = BlackScholes(OptionInputs(spot, strike, vol, rate, dte/365, kind, dividend))
    print(f"\nTheoretical premium: ${model.price():.4f} per share")
    print(f"Contract premium:    ${model.price()*CONTRACT_MULTIPLIER:.2f}")
    print(f"Risk-neutral ITM:    {model.probability_itm_risk_neutral():.2%}")
    print("Greeks:")
    for name, value in model.greeks().items():
        print(f"  {name.capitalize():8s}: {value:.6f}")
    print("Vega/Rho per 1 percentage point; Theta per calendar day.")


def payoff_plot():
    print("\nOPTION EXPIRATION PAYOFF")
    spot = float(input("Current underlying price [600]: ") or "600")
    strike = float(input("Strike [600]: ") or "600")
    premium = float(input("Premium paid per share [10]: ") or "10")
    kind = input("Call or put [call]: ").strip().lower() or "call"
    if kind not in ("call", "put"):
        raise ValueError("Choose call or put.")
    prices = np.linspace(spot*0.7, spot*1.3, 250)
    intrinsic = np.maximum(prices-strike, 0) if kind == "call" \
        else np.maximum(strike-prices, 0)
    pnl = (intrinsic-premium)*CONTRACT_MULTIPLIER
    breakeven = strike+premium if kind == "call" else strike-premium
    plt.figure(figsize=(10, 5))
    plt.plot(prices, pnl, label=f"Long {kind.upper()} — 1 contract")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(strike, linestyle="--", label=f"Strike ${strike:.2f}")
    plt.axvline(breakeven, linestyle=":", label=f"Breakeven ${breakeven:.2f}")
    plt.title(f"Long {kind.upper()} Expiration Payoff")
    plt.xlabel("Underlying at expiration")
    plt.ylabel("Profit / Loss (USD)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------- ORIGINAL GBM ANIMATED SIMULATOR -----------------

class LiveOptionsGraph:
    """Original three-panel GBM animation, retained with corrected GBM."""

    def __init__(self, S=64.5, K=65, sigma=0.40, r=0.06,
                 drift=0.20, expiry_days=30, option_type="call"):
        self.S, self.K = float(S), float(K)
        self.sigma, self.r, self.drift = float(sigma), float(r), float(drift)
        self.option_type = option_type.lower()
        self.total_days = max(1, int(expiry_days))
        self.remaining = self.total_days
        self.elapsed = 0
        self.dt = 1/TRADING_DAYS
        self.x, self.stock = [0], [self.S]
        self.option_prices, self.deltas = [], []
        if self.S <= 0 or self.K <= 0 or self.sigma <= 0:
            raise ValueError("S, K and volatility must be positive.")
        if self.option_type not in ("call", "put"):
            raise ValueError("Option type must be call or put.")

    def values(self):
        model = BlackScholes(OptionInputs(
            self.S, self.K, self.sigma, self.r,
            max(self.remaining, 0)/TRADING_DAYS, self.option_type
        ))
        return model.price(), model.greeks()["delta"], model.probability_itm_risk_neutral()

    def initialize(self):
        self.fig, self.ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        self.fig.suptitle(f"European {self.option_type.upper()} — GBM Simulation")
        self.ax[0].set_ylabel("Option premium ($)")
        self.ax[0].set_title("Option price")
        self.ax[1].set_ylabel("Delta")
        self.ax[1].set_title("Delta")
        self.ax[2].set_ylabel("Underlying ($)")
        self.ax[2].set_xlabel("Trading days")
        self.ax[2].set_title("Underlying vs strike")
        for axis in self.ax:
            axis.grid(alpha=0.3)
        self.l1, = self.ax[0].plot([], [], label="Option price")
        self.l2, = self.ax[1].plot([], [], label="Delta")
        self.l3, = self.ax[2].plot([], [], label="Underlying")
        self.ax[2].axhline(self.K, linestyle="--", label=f"Strike {self.K}")
        for axis in self.ax:
            axis.legend()
        self.status = self.fig.text(0.5, 0.01, "", ha="center", fontsize=9)
        self.fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(self, _frame):
        if self.remaining <= 0:
            return self.l1, self.l2, self.l3, self.status
        z = np.random.normal()
        self.S *= math.exp((self.drift-0.5*self.sigma**2)*self.dt
                           + self.sigma*math.sqrt(self.dt)*z)
        self.elapsed += 1
        self.remaining -= 1
        price, delta, prob = self.values()
        self.x.append(self.elapsed)
        self.stock.append(self.S)
        self.option_prices.append(price)
        self.deltas.append(delta)
        xx = np.arange(1, len(self.option_prices)+1)
        self.l1.set_data(xx, self.option_prices)
        self.l2.set_data(xx, self.deltas)
        self.l3.set_data(self.x, self.stock)
        for axis in (self.ax[0], self.ax[2]):
            axis.relim()
            axis.autoscale_view()
        self.ax[1].set_ylim(-1.05, 1.05)
        self.ax[1].set_xlim(0, max(self.total_days, 10))
        self.status.set_text(
            f"Day {self.elapsed}/{self.total_days} | Spot ${self.S:.2f} | "
            f"Option ${price:.4f} | Delta {delta:.4f} | "
            f"Risk-neutral ITM {prob:.1%} | DTE {self.remaining}"
        )
        return self.l1, self.l2, self.l3, self.status

    def run(self):
        self.initialize()
        self.anim = FuncAnimation(self.fig, self.update, frames=self.total_days,
                                  interval=500, repeat=False, blit=False,
                                  cache_frame_data=False)
        plt.show()


# -------------------------------- MENU -----------------------------

def main():
    alpaca = AlpacaDataClient()
    print("="*74)
    print("US OPTIONS ANALYTICS TERMINAL — ALPACA + YAHOO FINANCE")
    print("="*74)
    print(f"Alpaca credentials: {'configured' if alpaca.configured else 'not configured'}")
    print("Options feed: Alpaca indicative (where available); Yahoo fallback.")
    print("SPX index data: Yahoo Finance (^GSPC). SPY: Alpaca IEX + Yahoo fallback.")
    print("No orders are placed.")

    while True:
        print("\nMAIN MENU")
        print("1. Black-Scholes price + all five Greeks")
        print("2. SPY / SPX market monitor (Alpaca + Yahoo)")
        print("3. SPY / SPX option chain (Alpaca indicative + Yahoo fallback)")
        print("4. Download Alpaca historical indicative option bars (OCC symbol)")
        print("5. Synthetic Black-Scholes historical backtest")
        print("6. Option expiration payoff / breakeven chart")
        print("7. Original animated GBM simulator (three charts)")
        print("8. Exit")
        choice = input("\nSelect 1-8: ").strip()
        try:
            if choice == "1":
                interactive_pricer()
            elif choice == "2":
                market_monitor()
            elif choice == "3":
                show_option_chain()
            elif choice == "4":
                historical_indicative_option_data()
            elif choice == "5":
                historical_backtest()
            elif choice == "6":
                payoff_plot()
            elif choice == "7":
                kind = input("Call or put [call]: ").strip().lower() or "call"
                LiveOptionsGraph(S=64.5, K=65, sigma=0.40, r=0.06,
                                 drift=0.20, expiry_days=30,
                                 option_type=kind).run()
            elif choice == "8":
                print("Goodbye.")
                break
            else:
                print("Please choose a number from 1 to 8.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except Exception as exc:
            print(f"\nERROR: {exc}")
            print("Check inputs, connection, API keys, and data permissions.")


if __name__ == "__main__":
    main()