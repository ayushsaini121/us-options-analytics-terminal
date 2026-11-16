# US Options Analytics Terminal

### A Python-Based Quantitative Finance & Options Analytics Platform

Analyze US equity and index options with mathematical pricing models, market data integration, options Greeks, historical strategy backtesting, and risk visualization.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)](https://matplotlib.org/)
[![Market Data](https://img.shields.io/badge/Market%20Data-Alpaca%20%7C%20Yahoo-success)](https://alpaca.markets/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

---

## Overview

The **US Options Analytics Terminal** is a Python-based quantitative finance application designed to analyze options markets, evaluate derivatives pricing models, and explore systematic options trading strategies.

The platform combines financial mathematics, market data APIs, statistical simulation, and data visualization in an interactive terminal environment.

It currently focuses on two major US market instruments:

- **SPY** — SPDR S&P 500 ETF Trust
- **SPX** — S&P 500 Index

The application integrates Alpaca Market Data and Yahoo Finance to retrieve underlying market information and options data where available.

It also includes an analytical backtesting engine based on synthetic Black–Scholes option pricing.

## Key Features

### 1. Black–Scholes Options Pricing

Calculate theoretical European option prices using the Black–Scholes model.

- Call option valuation
- Put option valuation
- Time-to-expiration calculations
- Volatility and interest-rate assumptions
- Risk-neutral probability of expiring in the money

### 2. Options Greeks

Calculate all five primary options Greeks to understand the sensitivity of option prices to different market variables.

| Greek | Measures |
|---|---|
| Delta | Sensitivity to underlying price |
| Gamma | Rate of change of Delta |
| Theta | Sensitivity to time decay |
| Vega | Sensitivity to implied volatility |
| Rho | Sensitivity to interest rates |

### 3. US Market Monitor

Monitor SPY and SPX market information using:

- Alpaca Market Data API
- Yahoo Finance fallback
- Underlying market quotes
- Historical price information

### 4. Options Chain Analytics

Retrieve and examine options-chain information using available data providers.

- Calls and puts
- Strike prices
- Expiration dates
- Available option quotes and market information
- Alpaca indicative options snapshots
- Yahoo Finance fallback where available

### 5. Historical Options Data

Retrieve historical Alpaca indicative options bars using OCC option symbols.

This functionality provides a foundation for studying historical options price behaviour and developing more advanced research workflows.

### 6. Historical Strategy Backtesting

Evaluate systematic options strategies using historical underlying market data and synthetic Black–Scholes option valuations.

Current backtesting functionality includes:

- Historical underlying price data
- SMA-based trading signals
- Synthetic option valuation
- Entry and exit logic
- Transaction cost assumptions
- Portfolio equity tracking
- Backtesting performance summaries

**Important:** This is a synthetic options backtesting framework. It does not currently represent a complete historical execution simulation using historical options bid/ask quotes, order-book liquidity, and actual fills.

### 7. Options Payoff & Breakeven Analysis

Visualize option payoff profiles at expiration.

- Long call
- Long put
- Profit and loss analysis
- Breakeven calculations
- Strike-price analysis
- Maximum profit and maximum loss

### 8. Geometric Brownian Motion Simulation

Explore stochastic price movements through Geometric Brownian Motion (GBM).

The simulator generates stochastic price paths and visualizes simulated market behaviour through three charts.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| NumPy | Numerical computations |
| Pandas | Financial data processing |
| Matplotlib | Financial visualization |
| Requests | REST API integration |
| yfinance | Yahoo Finance market data |
| Alpaca API | Market and options data |

## Application Architecture

```text
                 US OPTIONS ANALYTICS TERMINAL
                              |
                Interactive Python Interface
                              |
          +-------------------+-------------------+
          |                   |                   |
     Market Data         Pricing Engine       Simulation
          |                   |                   |
    +-----+------+       Black-Scholes       Geometric
    |            |       Pricing Model       Brownian Motion
  Alpaca       Yahoo           |
  Markets     Finance       Greeks
    |            |             |
    +------------+-------------+
                 |
          Options Analytics
                 |
       +---------+----------+
       |         |          |
   Option     Historical   Payoff &
   Chains     Backtesting  Breakeven
```

## Installation & Setup

### Prerequisites

- Python 3.10 or later
- Git
- An Alpaca account for API-based market data access

### 1. Clone the repository

```bash
git clone https://github.com/ayushsaini121/us-options-analytics-terminal.git

cd us-options-analytics-terminal
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy pandas matplotlib requests yfinance
```

### 3. Configure Alpaca API credentials

Create an Alpaca account and obtain your API credentials.

Set the credentials as environment variables.

**Windows PowerShell:**

```powershell
$env:ALPACA_API_KEY="YOUR_API_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET_KEY"
```

Never commit API keys, secret keys, or `.env` files to GitHub.

### 4. Run the application

```bash
python us_options_analytics_terminal.py
```

## Application Menu

The terminal currently provides the following modules:

| Option | Functionality |
|---|---|
| 1 | Black–Scholes Pricing + Five Greeks |
| 2 | SPY / SPX Market Monitor |
| 3 | SPY / SPX Options Chain |
| 4 | Historical Indicative Options Bars |
| 5 | Synthetic Black–Scholes Backtesting |
| 6 | Options Expiration Payoff / Breakeven |
| 7 | Animated GBM Simulator |
| 8 | Exit |

## Market Data & Methodology

The application uses a combination of market data providers.

| Data Source | Usage |
|---|---|
| Alpaca | Equity market data and indicative options data |
| Yahoo Finance | Underlying historical prices and options data fallback |

### Data limitations

- Alpaca IEX equity data is not consolidated US market data.
- Alpaca indicative options data should not be interpreted as OPRA consolidated NBBO quotes.
- Yahoo Finance data availability and reliability may vary.
- Synthetic Black–Scholes option valuations are model-generated prices, not historical executable market prices.

These distinctions are important when interpreting analytics and backtesting results.

## Project Structure

```text
us-options-analytics-terminal/
│
├── us_options_analytics_terminal.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Development Roadmap

The project is under active development. Planned enhancements include:

- [ ] Implied volatility calculation
- [ ] Implied volatility surface visualization
- [ ] Monte Carlo option pricing
- [ ] Portfolio-level Greeks and exposure analytics
- [ ] Advanced performance metrics: Sharpe, Sortino, CAGR and profit factor
- [ ] Historical options bid/ask backtesting
- [ ] Walk-forward and out-of-sample strategy validation
- [ ] Transaction cost, slippage and liquidity modelling
- [ ] Strategy comparison against SPY buy-and-hold
- [ ] Automated unit testing
- [ ] Interactive analytics dashboard

## Disclaimer

This project is intended for educational, research, and quantitative finance development purposes.

It does not execute trades, provide investment advice, or guarantee the accuracy of theoretical valuations or backtesting results.

Options involve substantial financial risk. Synthetic backtesting results should not be interpreted as evidence of future profitability.

## Author

**Ayush Saini**

Quantitative Finance | Python | Financial Engineering | Data Science

GitHub: [@ayushsaini121](https://github.com/ayushsaini121)

---

### If you find this project interesting

Feel free to explore the repository, suggest improvements, and contribute to its development.

**Built with Python, financial mathematics, and quantitative research.**
