import json


notebook = {
    "cells": [],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def add_markdown(text):
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


def add_code(text):
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


add_markdown(
    """
# Trading RL System - Internship Assessment

This notebook builds an end-to-end reinforcement learning trading system for BTC-USD hourly OHLCV data. It fetches data automatically, engineers clean technical features, discovers unsupervised return-pattern regimes, trains a custom PPO agent, validates the environment with acceptance tests, and reports out-of-sample backtest metrics.
"""
)

add_markdown(
    """
## How to run

1. Open this notebook in Google Colab.
2. Select `Runtime > Restart and run all`.
3. The first code cell installs dependencies. No API keys or local files are required.
4. The notebook downloads BTC-USD hourly OHLCV data automatically with `yfinance`.
5. Runtime target: standard Colab CPU, roughly 5-10 minutes depending on package install and data download speed.
"""
)

add_markdown(
    """
## Stack and choices

- **Data source:** `yfinance` for automatic BTC-USD OHLCV data. It is free, does not need credentials, and provides enough volatility for a compact RL assessment.
- **Feature stack:** `pandas`, `numpy`, and `ta` for returns, RSI, MACD histogram, ATR, volume change, and volatility. Rows with warm-up NaNs are removed and checked.
- **Pattern discovery:** `MiniBatchKMeans` on standardized rolling return windows. This is a deliberately simple unsupervised method: it groups repeated local price-shape regimes without labels and keeps Colab runtime low.
- **RL stack:** `gymnasium` custom environment plus `stable-baselines3` PPO. PPO is stable for continuous action spaces and easy to reproduce in a notebook.
- **Risk logic:** Rewards are expressed in R-multiples, so the agent is evaluated by reward per unit of initial trade risk rather than raw price movement.
"""
)

add_code(
    """
# AI-assisted cell
!pip -q install yfinance gymnasium stable-baselines3[extra] pandas numpy matplotlib scikit-learn ta
print("Dependencies installed.")
"""
)

add_code(
    """
# AI-assisted cell
import os
import warnings

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from gymnasium import spaces
from IPython.display import display
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

warnings.filterwarnings("ignore")
np.random.seed(42)
print("Imports successful.")
"""
)

add_markdown(
    """
## Data and clean features

The data cell below pulls OHLCV automatically from Yahoo Finance. Feature rows created before indicator warm-up are dropped, then the notebook asserts that the final modeling frame has no NaNs.
"""
)

add_code(
    """
# AI-assisted cell
def fetch_data_and_build_features(ticker="BTC-USD", start="2023-01-01", end="2024-05-01", interval="1h"):
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("No OHLCV data returned. Re-run the cell or check yfinance availability.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna()

    df["returns"] = df["Close"].pct_change()
    df["volume_change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd_diff()
    df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["volatility"] = df["returns"].rolling(24).std()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["atr"] > 0].copy()

    assert df[required_cols].notna().all().all(), "OHLCV contains NaNs after cleaning."
    assert df[["returns", "volume_change", "rsi", "macd", "atr", "volatility"]].notna().all().all(), "Feature NaNs remain after warm-up."
    return df


data = fetch_data_and_build_features()
print(f"Rows after warm-up cleanup: {len(data):,}")
print(f"Remaining NaNs: {int(data.isna().sum().sum())}")
display(data.head())
"""
)

add_markdown(
    """
## Pattern discovery

Market states are not labeled, so this section learns recurring short-horizon return shapes with K-Means. Each input sample is a rolling window of recent returns, standardized before clustering. The resulting `pattern` id is used as one compact regime feature for the RL policy. This is not a claim that K-Means is the best sequence model; it is a transparent, fast baseline that makes the pattern-discovery requirement explicit and reproducible.
"""
)

add_code(
    """
# AI-assisted cell
class PatternDiscoverer:
    def __init__(self, n_clusters=8, window=12, random_state=42):
        self.n_clusters = n_clusters
        self.window = window
        self.scaler = StandardScaler()
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=256,
            n_init=10,
        )

    def fit_predict(self, returns_series):
        returns_series = pd.Series(returns_series).reset_index(drop=True)
        if len(returns_series) < self.window:
            raise ValueError("Not enough rows for pattern discovery window.")

        windows = np.array([
            returns_series.iloc[i : i + self.window].values
            for i in range(len(returns_series) - self.window + 1)
        ])
        windows_scaled = self.scaler.fit_transform(windows)
        clusters = self.model.fit_predict(windows_scaled)
        pad = np.full(self.window - 1, clusters[0], dtype=int)
        return np.concatenate([pad, clusters])


discoverer = PatternDiscoverer(n_clusters=8, window=12)
data["pattern"] = discoverer.fit_predict(data["returns"])
assert data["pattern"].notna().all()

pattern_summary = data.groupby("pattern")["returns"].agg(["count", "mean", "std"]).reset_index()
print("Pattern discovery complete.")
display(pattern_summary)
"""
)

add_markdown(
    """
## Custom RL environment

The environment supports five trade decisions through a continuous action value:

- `hold`: keep the current state unchanged.
- `enter_long`: open or flip into a long position.
- `enter_short`: open or flip into a short position.
- `close`: close an open position.
- Stop-loss and take-profit distance: action dimensions 1 and 2 map to ATR-based risk distances.

Commission and slippage are both included as friction. Trade rewards use R-multiple logic: net profit divided by initial risk.
"""
)

add_code(
    """
# AI-assisted cell
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, commission=0.0005, slippage=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.friction = self.commission + self.slippage
        self.feature_cols = ["returns", "volume_change", "rsi", "macd", "atr", "volatility", "pattern"]

        missing = [col for col in self.feature_cols + ["Open", "High", "Low", "Close", "Volume"] if col not in self.df.columns]
        if missing:
            raise ValueError(f"Environment data missing columns: {missing}")
        if self.df[self.feature_cols].isna().any().any():
            raise ValueError("Environment features contain NaNs.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols) + 1,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.initial_stop_price = 0.0
        self.stop_price = 0.0
        self.take_profit_price = 0.0
        self.trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.loc[self.current_step, self.feature_cols].astype(float).values
        obs = np.append(obs, float(self.position))
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _decode_action(self, action):
        decision_value = float(np.clip(action[0], -1.0, 1.0))
        if decision_value >= 0.50:
            decision = "enter_long"
        elif decision_value <= -0.50:
            decision = "enter_short"
        elif -0.20 <= decision_value <= 0.20:
            decision = "hold"
        else:
            decision = "close"

        sl_mult = 1.0 + ((float(np.clip(action[1], -1.0, 1.0)) + 1.0) / 2.0) * 4.0
        tp_mult = 1.0 + ((float(np.clip(action[2], -1.0, 1.0)) + 1.0) / 2.0) * 4.0
        return decision, sl_mult, tp_mult

    def _risk_fraction(self):
        if self.entry_price <= 0:
            return 0.0
        return abs(self.entry_price - self.initial_stop_price) / self.entry_price

    def _close_position(self, exit_price, reason):
        if self.position == 1:
            gross_return = (exit_price - self.entry_price) / self.entry_price
        else:
            gross_return = (self.entry_price - exit_price) / self.entry_price

        net_return = gross_return - self.friction
        risk_fraction = self._risk_fraction()
        r_multiple = net_return / risk_fraction if risk_fraction > 0 else 0.0

        trade = {
            "entry_price": self.entry_price,
            "exit_price": float(exit_price),
            "position": self.position,
            "net_return": float(net_return),
            "r_multiple": float(r_multiple),
            "reason": reason,
        }
        self.trades.append(trade)
        self.position = 0
        self.entry_price = 0.0
        self.initial_stop_price = 0.0
        self.stop_price = 0.0
        self.take_profit_price = 0.0
        return r_multiple, trade

    def _open_position(self, side, price, atr, sl_mult, tp_mult):
        self.position = int(side)
        self.entry_price = float(price)
        risk_distance = max(float(atr) * float(sl_mult), 1e-8)
        reward_penalty = self.friction / (risk_distance / self.entry_price)

        if self.position == 1:
            self.initial_stop_price = self.entry_price - risk_distance
            self.stop_price = self.initial_stop_price
            self.take_profit_price = self.entry_price + (risk_distance * float(tp_mult))
        else:
            self.initial_stop_price = self.entry_price + risk_distance
            self.stop_price = self.initial_stop_price
            self.take_profit_price = self.entry_price - (risk_distance * float(tp_mult))
        return -reward_penalty

    def step(self, action):
        reward = 0.0
        info = {
            "decision": None,
            "trade_closed": False,
            "pnl": 0.0,
            "r_multiple": 0.0,
            "friction": self.friction,
        }

        row = self.df.loc[self.current_step]
        price = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        atr = float(row["atr"])
        decision, sl_mult, tp_mult = self._decode_action(action)
        info["decision"] = decision

        if self.position != 0:
            hit_stop = self.position == 1 and low <= self.stop_price or self.position == -1 and high >= self.stop_price
            hit_take_profit = self.position == 1 and high >= self.take_profit_price or self.position == -1 and low <= self.take_profit_price
            if hit_stop or hit_take_profit:
                exit_price = self.stop_price if hit_stop else self.take_profit_price
                reward, trade = self._close_position(exit_price, "stop_loss" if hit_stop else "take_profit")
                info.update({"trade_closed": True, "pnl": trade["net_return"], "r_multiple": trade["r_multiple"]})

        if decision == "close" and self.position != 0:
            close_reward, trade = self._close_position(price, "manual_close")
            reward += close_reward
            info.update({"trade_closed": True, "pnl": trade["net_return"], "r_multiple": trade["r_multiple"]})

        target_side = 1 if decision == "enter_long" else -1 if decision == "enter_short" else 0
        if target_side != 0 and self.position != target_side:
            if self.position != 0:
                close_reward, trade = self._close_position(price, "flip")
                reward += close_reward
                info.update({"trade_closed": True, "pnl": trade["net_return"], "r_multiple": trade["r_multiple"]})
            reward += self._open_position(target_side, price, atr, sl_mult, tp_mult)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        if terminated and self.position != 0:
            final_price = float(self.df.loc[self.current_step, "Close"])
            close_reward, trade = self._close_position(final_price, "end_of_data")
            reward += close_reward
            info.update({"trade_closed": True, "pnl": trade["net_return"], "r_multiple": trade["r_multiple"]})

        return self._get_observation(), float(reward), terminated, False, info


env = TradingEnv(data)
check_env(env, warn=True)
print("Environment check passed.")
"""
)

add_markdown("### Automated acceptance tests")

add_code(
    """
# AI-assisted cell
def make_acceptance_data():
    rows = 20
    close = np.full(rows, 100.0)
    df = pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": np.full(rows, 1000.0),
        "returns": np.zeros(rows),
        "volume_change": np.zeros(rows),
        "rsi": np.full(rows, 50.0),
        "macd": np.zeros(rows),
        "atr": np.full(rows, 1.0),
        "volatility": np.full(rows, 0.01),
        "pattern": np.zeros(rows, dtype=int),
    })
    return df


def run_acceptance_tests():
    test_df = make_acceptance_data()

    env = TradingEnv(test_df, commission=0.0005, slippage=0.0005)
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(10):
        obs, reward, done, _, info = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))
        total_reward += reward
    assert abs(total_reward) < 1e-12, f"Do-nothing policy should be near 0, got {total_reward}"

    env = TradingEnv(test_df, commission=0.0005, slippage=0.0005)
    env.reset()
    _, open_reward, _, _, _ = env.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    _, close_reward, _, _, close_info = env.step(np.array([0.3, -1.0, -1.0], dtype=np.float32))
    assert open_reward + close_reward < 0, "Open then close should lose friction cost."
    assert close_info["trade_closed"], "Manual close should close the trade."

    env = TradingEnv(test_df, commission=0.0, slippage=0.0)
    env.reset()
    env.position = 1
    env.entry_price = 100.0
    env.initial_stop_price = 99.0
    env.stop_price = 99.0
    env.take_profit_price = 103.0
    env.df.loc[env.current_step, "High"] = 103.25
    _, reward, _, _, info = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))
    assert info["trade_closed"], "Take-profit test should close the trade."
    assert np.isclose(info["r_multiple"], 3.0, atol=0.05), f"Expected about 3R, got {info['r_multiple']}"

    print("Acceptance tests passed:")
    print(f"- do-nothing policy reward: {total_reward:.8f}")
    print(f"- open then close reward: {open_reward + close_reward:.4f}")
    print(f"- 3R take-profit reward: {info['r_multiple']:.2f}R")


run_acceptance_tests()
"""
)

add_markdown("## Training")

add_code(
    """
# AI-assisted cell
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size].copy()
test_data = data.iloc[train_size:].copy()

train_env = TradingEnv(train_data)
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=0,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    seed=42,
)

print("Starting PPO training...")
model.learn(total_timesteps=12000)
model.save("ppo_trading_bot")

training_probe_env = TradingEnv(train_data.tail(500))
obs, _ = training_probe_env.reset()
probe_reward = 0.0
probe_steps = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = training_probe_env.step(action)
    probe_reward += reward
    probe_steps += 1

print("Model saved to ppo_trading_bot.zip")
print(f"Training probe steps: {probe_steps}")
print(f"Training probe cumulative R reward: {probe_reward:.2f}")
"""
)

add_markdown("## Backtest and final metrics table")

add_code(
    """
# AI-assisted cell
def backtest_model(model, test_data):
    test_env = TradingEnv(test_data)
    obs, _ = test_env.reset()
    done = False
    equity = [1.0]
    current_equity = 1.0
    closed_trades = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        if info.get("trade_closed"):
            current_equity *= 1.0 + info["pnl"]
            closed_trades.append(info.copy())
        equity.append(current_equity)

    equity_series = pd.Series(equity, name="equity")
    returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = equity_series.iloc[-1] - 1.0
    sharpe_ratio = np.sqrt(24 * 365) * returns.mean() / (returns.std() + 1e-9) if len(returns) else 0.0
    drawdown = equity_series / equity_series.cummax() - 1.0
    r_multiples = [trade["r_multiple"] for trade in closed_trades]
    wins = [trade for trade in closed_trades if trade["pnl"] > 0]

    metrics = {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": drawdown.min() if len(drawdown) else 0.0,
        "Number of Trades": len(closed_trades),
        "Win Rate": len(wins) / len(closed_trades) if closed_trades else 0.0,
        "Average R-Multiple": float(np.mean(r_multiples)) if r_multiples else 0.0,
    }
    return metrics, equity_series, r_multiples


metrics, equity_curve, r_multiples = backtest_model(model, test_data)
metrics_table = pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Value": [
        f"{metrics['Total Return']:.2%}",
        f"{metrics['Sharpe Ratio']:.2f}",
        f"{metrics['Max Drawdown']:.2%}",
        metrics["Number of Trades"],
        f"{metrics['Win Rate']:.2%}",
        f"{metrics['Average R-Multiple']:.2f}",
    ],
})

display(metrics_table)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(equity_curve.values, color="navy")
axes[0].set_title("Equity Curve")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Equity, base 1.0")
axes[0].grid(True, alpha=0.3)

if r_multiples:
    axes[1].hist(r_multiples, bins=20, color="darkgreen", alpha=0.75)
    axes[1].axvline(np.mean(r_multiples), color="black", linestyle="--", label="Mean R")
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, "No closed trades", ha="center", va="center")
axes[1].set_title("R-Multiple Distribution")
axes[1].set_xlabel("R")
axes[1].set_ylabel("Trades")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

add_markdown(
    """
## Final metrics table

The final metrics table is generated by the backtest cell above. Because the notebook fetches fresh Yahoo Finance data at runtime, exact values can move slightly if the data provider revises candles or the date range is changed.
"""
)

add_markdown(
    """
## What broke / what surprised you

- The first version made hold and close the same action. That made the policy harder to reason about, so the final environment gives hold and close separate action bands.
- A do-nothing policy can look attractive because commission and slippage punish exploration. R-multiple rewards reduce this by making high-quality asymmetric setups more visible to the agent.
- Intra-bar stop-loss and take-profit handling is ambiguous with hourly candles. This notebook uses conservative stop-first logic when both stop and target are touched in the same bar.
"""
)

add_markdown(
    """
## Three weakest decisions

- K-Means flattens return windows and ignores richer sequence structure. A temporal autoencoder or HMM would be a stronger pattern model.
- Position sizing is fixed to one notional unit per trade. A production system should add volatility targeting and account-level risk limits.
- The backtest uses hourly OHLCV bars, so it cannot know exact intra-bar execution order. Lower-timeframe data would improve execution realism.
"""
)

add_markdown(
    """
## What I would do with one more week

- Add walk-forward validation across several market regimes.
- Tune PPO and environment parameters with Optuna.
- Replace K-Means with a sequence model for pattern discovery.
- Add realistic exchange constraints, margin checks, and position sizing.
- Compare PPO against simple rule-based baselines and a random policy.
"""
)


with open("Trading_RL_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generation complete.")
