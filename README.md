# Trading RL System

Internship assessment project for building a compact reinforcement learning trading system in Google Colab.

## Overview

This repository contains an end-to-end notebook that:

- Pulls BTC-USD hourly OHLCV data automatically with `yfinance`.
- Builds clean technical features after indicator warm-up.
- Adds unsupervised pattern discovery with K-Means over rolling return windows.
- Implements a custom `gymnasium` trading environment.
- Trains a PPO agent with `stable-baselines3`.
- Evaluates the strategy with backtest metrics and inline plots.
- Includes automated acceptance tests for the core reward and execution logic.

## Features

- Automatic OHLCV data download, no API key required.
- Feature checks to ensure no NaNs remain after warm-up.
- Continuous action space with explicit support for:
  - hold
  - enter long
  - enter short
  - close
  - ATR-based stop-loss distance
  - ATR-based take-profit distance
- Commission and slippage included as trading friction.
- R-multiple reward logic based on net return divided by initial trade risk.
- Acceptance tests for:
  - do-nothing policy near-zero reward
  - open then close losing friction cost
  - 3R take-profit producing about +3 reward
- Backtest metrics:
  - total return
  - Sharpe ratio
  - max drawdown
  - number of trades
  - win rate
  - average R-multiple

## Tech stack

- Python
- Google Colab
- `pandas`, `numpy`
- `yfinance`
- `ta`
- `scikit-learn`
- `gymnasium`
- `stable-baselines3`
- `matplotlib`

## How to run in Colab

1. Open `Trading_RL_System.ipynb` in Google Colab.
2. Select `Runtime > Restart and run all`.
3. The first code cell installs all notebook dependencies.
4. The notebook downloads BTC-USD data automatically.
5. Review the final metrics table, equity curve, and R-multiple distribution at the end.

## Local acceptance tests

The local tests use synthetic data, so they do not require network access:

```bash
python3 acceptance_tests.py
```

Expected output:

```text
All acceptance tests passed.
```

## Metrics

Final backtest metrics are printed inside the notebook after training. They are generated at runtime because the notebook pulls market data automatically and trains the PPO model in-session.

Reported metrics include total return, Sharpe ratio, max drawdown, number of trades, win rate, and average R-multiple.

## Limitations

- K-Means pattern discovery is a simple baseline and does not model sequence dynamics as well as an HMM, temporal CNN, or recurrent autoencoder.
- Hourly OHLCV data cannot fully reconstruct intra-bar execution order.
- Position sizing is simplified to one notional unit per trade.
- PPO training is intentionally short so the notebook remains practical for Colab assessment review.
- Results should be treated as a research prototype, not live-trading advice.

## Submission note

This repository is prepared for the Trading RL System internship assessment. The notebook is designed to run top-to-bottom in Google Colab and includes the required markdown discussion sections, environment validation, acceptance tests, model saving, backtest metrics, and inline plots.
