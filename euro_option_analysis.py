
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# US OPTIONS ANALYTICS SIMULATOR - VERSION 2.0
# European Black-Scholes Model
# ============================================================


# -------------------- NORMAL DISTRIBUTION --------------------

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# -------------------- BLACK-SCHOLES MODEL --------------------

class EuropeanOption:

    def __init__(self, S, K, sigma, r, T, q=0.0,
                 option_type="call"):

        self.S = float(S)
        self.K = float(K)
        self.sigma = float(sigma)
        self.r = float(r)
        self.T = float(T)
        self.q = float(q)
        self.option_type = option_type.lower()

        if self.S <= 0 or self.K <= 0:
            raise ValueError("Stock and strike prices must be positive.")

        if self.sigma <= 0:
            raise ValueError("Volatility must be positive.")

        if self.T < 0:
            raise ValueError("Time to expiry cannot be negative.")

        if self.option_type not in ("call", "put"):
            raise ValueError("Option type must be call or put.")

    def d1(self):

        if self.T <= 0:
            return None

        return (
            math.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * math.sqrt(self.T))

    def d2(self):

        d1 = self.d1()

        if d1 is None:
            return None

        return d1 - self.sigma * math.sqrt(self.T)

    # -------------------- OPTION PRICE --------------------

    def price(self):

        if self.T == 0:

            if self.option_type == "call":
                return max(self.S - self.K, 0)

            return max(self.K - self.S, 0)

        d1 = self.d1()
        d2 = self.d2()

        if self.option_type == "call":

            return (
                self.S * math.exp(-self.q * self.T) * norm_cdf(d1)
                - self.K * math.exp(-self.r * self.T) * norm_cdf(d2)
            )

        return (
            self.K * math.exp(-self.r * self.T) * norm_cdf(-d2)
            - self.S * math.exp(-self.q * self.T) * norm_cdf(-d1)
        )

    # -------------------- DELTA --------------------

    def delta(self):

        if self.T == 0:

            if self.option_type == "call":
                return 1.0 if self.S > self.K else (
                    0.5 if self.S == self.K else 0.0
                )

            return -1.0 if self.S < self.K else (
                -0.5 if self.S == self.K else 0.0
            )

        d1 = self.d1()
        discount = math.exp(-self.q * self.T)

        if self.option_type == "call":
            return discount * norm_cdf(d1)

        return discount * (norm_cdf(d1) - 1)

    # -------------------- GAMMA --------------------

    def gamma(self):

        if self.T == 0:
            return 0.0

        return (
            math.exp(-self.q * self.T)
            * norm_pdf(self.d1())
            / (self.S * self.sigma * math.sqrt(self.T))
        )

    # -------------------- VEGA --------------------

    def vega(self):

        if self.T == 0:
            return 0.0

        value = (
            self.S
            * math.exp(-self.q * self.T)
            * norm_pdf(self.d1())
            * math.sqrt(self.T)
        )

        # Change for a 1 percentage point change in volatility
        return value / 100

    # -------------------- THETA --------------------

    def theta(self):

        if self.T == 0:
            return 0.0

        d1 = self.d1()
        d2 = self.d2()

        first_term = (
            -self.S * math.exp(-self.q * self.T)
            * norm_pdf(d1) * self.sigma
            / (2 * math.sqrt(self.T))
        )

        if self.option_type == "call":

            annual_theta = (
                first_term
                - self.r * self.K * math.exp(-self.r * self.T)
                * norm_cdf(d2)
                + self.q * self.S * math.exp(-self.q * self.T)
                * norm_cdf(d1)
            )

        else:

            annual_theta = (
                first_term
                + self.r * self.K * math.exp(-self.r * self.T)
                * norm_cdf(-d2)
                - self.q * self.S * math.exp(-self.q * self.T)
                * norm_cdf(-d1)
            )

        # Calendar-day theta
        return annual_theta / 365

    # -------------------- RHO --------------------

    def rho(self):

        if self.T == 0:
            return 0.0

        d2 = self.d2()

        if self.option_type == "call":

            value = (
                self.K * self.T * math.exp(-self.r * self.T)
                * norm_cdf(d2)
            )

        else:

            value = (
                -self.K * self.T * math.exp(-self.r * self.T)
                * norm_cdf(-d2)
            )

        # Change for a 1 percentage point change in interest rate
        return value / 100

    # -------------------- PROBABILITY ITM --------------------

    def probability_itm(self):

        if self.T == 0:

            if self.option_type == "call":
                return 1.0 if self.S > self.K else 0.0

            return 1.0 if self.S < self.K else 0.0

        if self.option_type == "call":
            return norm_cdf(self.d2())

        return norm_cdf(-self.d2())

    def all_greeks(self):

        return {
            "Delta": self.delta(),
            "Gamma": self.gamma(),
            "Theta/day": self.theta(),
            "Vega/1%": self.vega(),
            "Rho/1%": self.rho()
        }


# ============================================================
# IMPLIED VOLATILITY - BISECTION METHOD
# ============================================================

def calculate_implied_volatility(
    market_price, S, K, r, T, q=0.0,
    option_type="call"
):

    if market_price <= 0 or T <= 0:
        return None

    intrinsic = max(
        S * math.exp(-q * T) - K * math.exp(-r * T), 0
    ) if option_type == "call" else max(
        K * math.exp(-r * T) - S * math.exp(-q * T), 0
    )

    if market_price < intrinsic - 1e-8:
        return None

    low = 0.0001
    high = 5.0

    for _ in range(150):

        mid = (low + high) / 2

        option = EuropeanOption(
            S, K, mid, r, T, q, option_type
        )

        estimated_price = option.price()

        if estimated_price > market_price:
            high = mid
        else:
            low = mid

    return (low + high) / 2


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_pricing(
    S, K, sigma, r, T, q=0.0,
    option_type="call", simulations=50000
):

    rng = np.random.default_rng(42)

    Z = rng.standard_normal(simulations)

    terminal_prices = S * np.exp(
        (r - q - 0.5 * sigma**2) * T
        + sigma * math.sqrt(T) * Z
    )

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)

    discounted_payoffs = math.exp(-r * T) * payoffs

    estimated_price = np.mean(discounted_payoffs)

    standard_error = (
        np.std(discounted_payoffs, ddof=1)
        / math.sqrt(simulations)
    )

    return (
        estimated_price,
        standard_error,
        terminal_prices
    )


# ============================================================
# OPTIONS PAYOFF ANALYSIS
# ============================================================

def calculate_payoff_curve(
    S, K, premium, option_type, contracts=1,
    multiplier=100
):

    prices = np.linspace(
        max(0.01, S * 0.5),
        S * 1.5,
        200
    )

    if option_type == "call":
        payoff = np.maximum(prices - K, 0) - premium
        breakeven = K + premium

    else:
        payoff = np.maximum(K - prices, 0) - premium
        breakeven = K - premium

    pnl = payoff * contracts * multiplier

    return prices, pnl, breakeven


# ============================================================
# LIVE SIMULATOR
# ============================================================

class LiveOptionsGraph:

    def __init__(
        self,
        symbol,
        S,
        K,
        sigma,
        r,
        q,
        expiry_days,
        drift,
        option_type,
        contracts=1,
        multiplier=100,
        market_option_price=None
    ):

        self.symbol = symbol
        self.S = S
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.expiry_days = expiry_days
        self.drift = drift
        self.option_type = option_type.lower()
        self.contracts = contracts
        self.multiplier = multiplier
        self.market_option_price = market_option_price

        if self.S <= 0 or self.K <= 0:
            raise ValueError("Stock and strike prices must be positive.")

        if self.sigma <= 0:
            raise ValueError("Volatility must be positive.")

        if self.expiry_days <= 0:
            raise ValueError("Expiry days must be positive.")

        if self.contracts < 1:
            raise ValueError("Contracts must be at least 1.")

        if self.option_type not in ("call", "put"):
            raise ValueError("Select call or put.")

        self.initial_T = expiry_days / 365
        self.remaining_T = self.initial_T

        # Approximate number of US trading sessions
        self.total_steps = max(
            1, round(expiry_days * 252 / 365)
        )

        self.dt = 1 / 252
        self.elapsed_days = 0

        self.stock_history = [S]
        self.time_history = [0]

        initial_option = self.get_option(S, self.remaining_T)

        self.option_history = [initial_option.price()]
        self.delta_history = [initial_option.delta()]

        self.fig = None
        self.animation = None

    def get_option(self, S, T):

        return EuropeanOption(
            S, self.K, self.sigma,
            self.r, T, self.q, self.option_type
        )

    # -------------------- STOCK SIMULATION --------------------

    def simulate_stock_price(self):

        Z = np.random.normal()

        self.S *= math.exp(
            (self.drift - self.q - 0.5 * self.sigma**2) * self.dt
            + self.sigma * math.sqrt(self.dt) * Z
        )

    # -------------------- BUILD DASHBOARD --------------------

    def initialize_plot(self):

        self.fig, self.axes = plt.subplots(
            2, 3, figsize=(16, 9)
        )

        self.fig.suptitle(
            f"{self.symbol} | US OPTIONS ANALYTICS TERMINAL",
            fontsize=17,
            fontweight="bold"
        )

        ax1, ax2, ax3, ax4, ax5, ax6 = self.axes.flatten()

        # Animated chart 1
        ax1.set_title("Option Premium")
        ax1.set_xlabel("Trading Sessions")
        ax1.set_ylabel("Premium ($)")
        ax1.grid(True, alpha=0.3)

        self.option_line, = ax1.plot(
            [], [], linewidth=2, label="Option Price"
        )

        # Animated chart 2
        ax2.set_title("Option Delta")
        ax2.set_xlabel("Trading Sessions")
        ax2.set_ylabel("Delta")
        ax2.grid(True, alpha=0.3)

        self.delta_line, = ax2.plot(
            [], [], linewidth=2, label="Delta"
        )

        # Animated chart 3
        ax3.set_title("Underlying Price vs Strike")
        ax3.set_xlabel("Trading Sessions")
        ax3.set_ylabel("Stock Price ($)")
        ax3.grid(True, alpha=0.3)

        self.stock_line, = ax3.plot(
            [], [], linewidth=2, label="Underlying"
        )

        ax3.axhline(
            self.K, linestyle="--",
            label=f"Strike ${self.K:.2f}"
        )

        # Payoff chart
        ax4.set_title("Expiration P&L")
        ax4.set_xlabel("Underlying Price at Expiration ($)")
        ax4.set_ylabel("Profit / Loss ($)")
        ax4.grid(True, alpha=0.3)

        self.initial_premium = self.option_history[0]

        prices, pnl, breakeven = calculate_payoff_curve(
            self.S, self.K, self.initial_premium,
            self.option_type, self.contracts, self.multiplier
        )

        ax4.plot(prices, pnl, linewidth=2, label="P&L")
        ax4.axhline(0, linestyle="--")
        ax4.axvline(
            breakeven, linestyle=":",
            label=f"Breakeven ${breakeven:.2f}"
        )

        ax4.legend()

        # Monte Carlo chart
        ax5.set_title("Monte Carlo Terminal Price Distribution")
        ax5.set_xlabel("Terminal Price ($)")
        ax5.set_ylabel("Frequency")
        ax5.grid(True, alpha=0.2)

        mc_price, mc_error, terminal_prices = monte_carlo_pricing(
            self.S, self.K, self.sigma,
            self.r, self.initial_T, self.q,
            self.option_type
        )

        ax5.hist(
            terminal_prices,
            bins=50,
            alpha=0.7,
            edgecolor="black"
        )

        ax5.axvline(
            self.K, linestyle="--",
            label=f"Strike ${self.K:.2f}"
        )

        ax5.legend()

        # Greeks chart
        ax6.set_title("Initial Option Greeks")
        ax6.set_xlabel("Greek Value")
        ax6.grid(True, alpha=0.3)

        initial_greeks = self.get_option(
            self.S, self.initial_T
        ).all_greeks()

        names = list(initial_greeks.keys())
        values = list(initial_greeks.values())

        positions = np.arange(len(names))

        ax6.barh(positions, values)
        ax6.set_yticks(positions)
        ax6.set_yticklabels(names)
        ax6.invert_yaxis()

        self.fig.tight_layout(rect=[0, 0.07, 1, 0.94])

        self.status_text = self.fig.text(
            0.5, 0.015, "",
            ha="center",
            fontsize=9
        )

        self.mc_price = mc_price
        self.mc_error = mc_error

    # -------------------- UPDATE ANIMATION --------------------

    def update(self, frame):

        if self.remaining_T <= 0:
            return (
                self.option_line,
                self.delta_line,
                self.stock_line,
                self.status_text
            )

        self.simulate_stock_price()

        self.elapsed_days += 1

        self.remaining_T = max(
            0, self.initial_T - self.elapsed_days / 252
        )

        option = self.get_option(
            self.S, self.remaining_T
        )

        option_price = option.price()
        delta = option.delta()
        probability = option.probability_itm()

        self.stock_history.append(self.S)
        self.option_history.append(option_price)
        self.delta_history.append(delta)
        self.time_history.append(self.elapsed_days)

        self.option_line.set_data(
            self.time_history, self.option_history
        )

        self.delta_line.set_data(
            self.time_history, self.delta_history
        )

        self.stock_line.set_data(
            self.time_history, self.stock_history
        )

        for axis in self.axes[0, :3]:
            axis.relim()
            axis.autoscale_view()

        self.axes[0, 1].set_ylim(-1.05, 1.05)

        self.status_text.set_text(
            f"Underlying: ${self.S:.2f} | "
            f"Option: ${option_price:.4f} | "
            f"Delta: {delta:.4f} | "
            f"Probability ITM: {probability:.2%} | "
            f"Days remaining: {self.remaining_T * 365:.1f}"
        )

        return (
            self.option_line,
            self.delta_line,
            self.stock_line,
            self.status_text
        )

    # -------------------- RUN --------------------

    def run(self):

        self.initialize_plot()

        self.animation = FuncAnimation(
            self.fig,
            self.update,
            frames=self.total_steps,
            interval=500,
            blit=False,
            repeat=False,
            cache_frame_data=False
        )

        plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("       US OPTIONS ANALYTICS SIMULATOR - VERSION 2.0")
    print("=" * 65)

    # --------------------------------------------------------
    # USER INPUTS - CHANGE THESE VALUES
    # --------------------------------------------------------

    SYMBOL = "US INDEX (EXAMPLE)"

    STOCK_PRICE = 100.00
    STRIKE_PRICE = 100.00

    VOLATILITY = 0.25
    RISK_FREE_RATE = 0.04
    DIVIDEND_YIELD = 0.01

    EXPIRY_DAYS = 30
    DRIFT = 0.08

    OPTION_TYPE = "call"

    CONTRACTS = 1
    CONTRACT_MULTIPLIER = 100

    # Enter a known market option premium to calculate IV.
    # Leave None if you do not have a market quote.
    MARKET_OPTION_PRICE = None

    # --------------------------------------------------------
    # CREATE OPTION
    # --------------------------------------------------------

    option = EuropeanOption(
        S=STOCK_PRICE,
        K=STRIKE_PRICE,
        sigma=VOLATILITY,
        r=RISK_FREE_RATE,
        T=EXPIRY_DAYS / 365,
        q=DIVIDEND_YIELD,
        option_type=OPTION_TYPE
    )

    premium = option.price()
    greeks = option.all_greeks()

    print(f"\nUnderlying: {SYMBOL}")
    print(f"Option Type: {OPTION_TYPE.upper()}")
    print(f"Stock Price: ${STOCK_PRICE:.2f}")
    print(f"Strike Price: ${STRIKE_PRICE:.2f}")
    print(f"Volatility: {VOLATILITY:.2%}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE:.2%}")
    print(f"Dividend Yield: {DIVIDEND_YIELD:.2%}")
    print(f"Expiry: {EXPIRY_DAYS} calendar days")

    print("\n--- BLACK-SCHOLES RESULTS ---")

    print(f"Option Premium: ${premium:.4f}")
    print(f"Premium per Contract: ${premium * CONTRACT_MULTIPLIER:.2f}")
    print(f"Probability ITM: {option.probability_itm():.2%}")

    print("\n--- OPTION GREEKS ---")

    for name, value in greeks.items():
        print(f"{name}: {value:.6f}")

    # --------------------------------------------------------
    # IMPLIED VOLATILITY
    # --------------------------------------------------------

    if MARKET_OPTION_PRICE is not None:

        iv = calculate_implied_volatility(
            MARKET_OPTION_PRICE,
            STOCK_PRICE,
            STRIKE_PRICE,
            RISK_FREE_RATE,
            EXPIRY_DAYS / 365,
            DIVIDEND_YIELD,
            OPTION_TYPE
        )

        if iv is None:
            print("\nImplied Volatility: Could not calculate.")
        else:
            print(f"\nImplied Volatility: {iv:.2%}")

    else:
        print("\nImplied Volatility: Enter a market premium to calculate.")

    # --------------------------------------------------------
    # MONTE CARLO RESULTS
    # --------------------------------------------------------

    mc_price, mc_error, _ = monte_carlo_pricing(
        STOCK_PRICE,
        STRIKE_PRICE,
        VOLATILITY,
        RISK_FREE_RATE,
        EXPIRY_DAYS / 365,
        DIVIDEND_YIELD,
        OPTION_TYPE,
        simulations=50000
    )

    print("\n--- MONTE CARLO SIMULATION ---")

    print(f"Monte Carlo Price: ${mc_price:.4f}")
    print(f"Standard Error: ${mc_error:.4f}")
    print(f"Black-Scholes Price: ${premium:.4f}")
    print(f"Price Difference: ${mc_price - premium:.4f}")

    # --------------------------------------------------------
    # PAYOFF RESULTS
    # --------------------------------------------------------

    prices, pnl, breakeven = calculate_payoff_curve(
        STOCK_PRICE,
        STRIKE_PRICE,
        premium,
        OPTION_TYPE,
        CONTRACTS,
        CONTRACT_MULTIPLIER
    )

    print("\n--- EXPIRATION PAYOFF ---")

    print(f"Breakeven Price: ${breakeven:.2f}")
    print(f"Initial Premium Paid: ${premium * CONTRACTS * CONTRACT_MULTIPLIER:.2f}")

    if OPTION_TYPE == "call":
        print("Maximum Loss: Premium Paid")
        print("Maximum Profit: Unlimited")

    else:
        print("Maximum Loss: Limited by underlying falling to zero")
        print("Maximum Profit: Strike less premium, per share")

    print("\nStarting animated dashboard...")
    print("=" * 65)

    # --------------------------------------------------------
    # START SIMULATOR
    # --------------------------------------------------------

    simulator = LiveOptionsGraph(
        symbol=SYMBOL,
        S=STOCK_PRICE,
        K=STRIKE_PRICE,
        sigma=VOLATILITY,
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
        expiry_days=EXPIRY_DAYS,
        drift=DRIFT,
        option_type=OPTION_TYPE,
        contracts=CONTRACTS,
        multiplier=CONTRACT_MULTIPLIER,
        market_option_price=MARKET_OPTION_PRICE
    )

    simulator.run()
