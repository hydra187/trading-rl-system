# Trading RL System

Internship assessment repository for a compact reinforcement learning trading
system that runs in Google Colab.

The project is intentionally notebook-first. It downloads market data, builds
features, discovers simple market patterns, trains a PPO agent, runs acceptance
tests, and reports backtest metrics with plots.

## Project Overview

`Trading_RL_System.ipynb` is the main submission artifact. It is designed to run
top-to-bottom on a fresh Colab runtime.

The notebook performs the full workflow:

- installs dependencies
- downloads BTC-USD OHLCV data with `yfinance`
- engineers clean trading features
- adds unsupervised pattern-discovery labels
- defines a custom `gymnasium` RL trading environment
- trains a PPO policy with `stable-baselines3`
- saves the trained model
- runs acceptance tests for reward and execution logic
- reports backtest metrics and inline plots

## Repository Contents

| File | Purpose |
| --- | --- |
| `Trading_RL_System.ipynb` | Main Colab notebook for the assessment |
| `generate_notebook.py` | Script used to regenerate the notebook structure |
| `acceptance_tests.py` | Local synthetic-data tests for core environment behavior |
| `README.md` | Project documentation and submission notes |

`test_run.py` was removed because it was an unnecessary duplicate-style script
and not part of the clean submission.

## Features

- Automatic OHLCV data download.
- No API key required.
- Indicator warm-up cleanup with NaN checks.
- Unsupervised pattern discovery using rolling return windows.
- Custom RL environment with explicit trading actions.
- Commission and slippage included as friction.
- R-multiple reward logic.
- PPO training and model saving.
- Backtest metrics and inline plots.
- Local acceptance tests that run without network access.

## Tech Stack

- Python
- Google Colab
- `pandas`
- `numpy`
- `yfinance`
- `ta`
- `scikit-learn`
- `gymnasium`
- `stable-baselines3`
- `matplotlib`

## RL Architecture

The notebook uses a custom `gymnasium` environment and trains a PPO agent from
`stable-baselines3`.

### Observation Space

The observation includes:

- recent return
- volume change
- RSI
- MACD histogram
- ATR
- rolling volatility
- discovered pattern id
- current position state

### Action Space

The continuous action has three components:

- `action[0]`: trade decision
- `action[1]`: stop-loss distance
- `action[2]`: take-profit distance

The trade decision supports:

- hold
- enter long
- enter short
- close

Stop-loss and take-profit distances are mapped from the action values to
ATR-based distances.

## Pattern Discovery Approach

The pattern discovery step uses `MiniBatchKMeans` on standardized rolling
windows of returns.

Each rolling window represents a short local price-shape regime. The fitted
cluster id is added back to the dataset as a compact market-pattern feature for
the RL policy.

This approach is simple and explainable. It is not the strongest possible
sequence model, but it keeps the assessment lightweight and reproducible in
Colab.

## Reward System

The environment rewards closed trades using R-multiple logic:

```text
R-multiple = net trade return / initial trade risk
```

Initial trade risk is the distance between the entry price and initial
stop-loss. Net trade return includes commission and slippage.

This makes the reward more risk-aware than raw PnL because the agent is judged
by profit relative to the risk it accepted.

## Environment Validation

The environment includes:

- hold logic
- enter-long logic
- enter-short logic
- manual close logic
- stop-loss logic
- take-profit logic
- commission
- slippage

The local test file validates three important cases:

- do-nothing policy produces approximately zero reward
- instant open-close loses friction cost
- take-profit hit gives approximately the correct R-multiple reward

## Backtesting Metrics

The notebook reports:

- total return
- Sharpe ratio
- max drawdown
- number of trades
- win rate
- average R-multiple

The notebook also displays:

- equity curve
- R-multiple distribution plot

## How To Run In Colab

1. Open `Trading_RL_System.ipynb` in Google Colab.
2. Select `Runtime > Restart and run all`.
3. Let the first code cell install dependencies.
4. Confirm BTC-USD data downloads successfully.
5. Confirm acceptance tests pass inside the notebook.
6. Confirm the final metrics table and plots render at the end.

## Local Acceptance Tests

The local tests use synthetic data and do not require internet access.

Run:

```bash
python3 acceptance_tests.py
```

Expected output:

```text
All acceptance tests passed.
```

## Results

No fixed numeric results are hardcoded in this repository.

The final metrics are generated when the notebook runs because the workflow
downloads market data and trains the PPO model at runtime. This avoids
inventing or freezing results that may not reproduce exactly on a fresh Colab
runtime.

## Limitations

- K-Means is a simple pattern-discovery baseline and does not model sequence
  dynamics as well as a temporal model.
- Hourly OHLCV bars cannot fully reconstruct intra-bar execution order.
- Position sizing is simplified to one notional unit per trade.
- PPO training is intentionally short so the notebook remains practical for
  assessment review.
- This is a research prototype, not live-trading advice.

## Future Improvements

- Add walk-forward validation across multiple market regimes.
- Compare PPO against random, buy-and-hold, and rule-based baselines.
- Tune PPO and environment parameters with Optuna.
- Replace K-Means with a sequence-aware pattern model.
- Add position sizing, leverage limits, and exchange-specific execution rules.
- Test on lower-timeframe data to reduce intra-bar ambiguity.

## Submission Note

This repository is prepared for the Trading RL System internship assessment.

Before submitting, run the notebook once in a fresh Colab runtime and use the
metrics produced by that run. Do not report numeric results that were not
produced by the notebook.
