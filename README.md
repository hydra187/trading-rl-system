# Trading RL System

Internship assessment project for a reinforcement learning trading system.

The main deliverable is `Trading_RL_System.ipynb`.
It is designed to run top-to-bottom in Google Colab.

## Overview

This project builds a compact RL trading workflow.

It downloads market data automatically.
It builds technical features.
It discovers simple market patterns.
It trains a PPO reinforcement learning agent.
It evaluates the trained policy with backtest metrics and plots.

No fixed numeric results are hardcoded.
Final results should come from running the notebook.

## Repository Files

| File | Purpose |
| --- | --- |
| `Trading_RL_System.ipynb` | Main Colab notebook |
| `acceptance_tests.py` | Local acceptance tests |
| `generate_notebook.py` | Notebook generation helper |
| `README.md` | Project documentation |

`test_run.py` was removed.
It was not needed for the final submission.

## Features

- Automatic BTC-USD OHLCV data download.
- Clean feature engineering after indicator warm-up.
- NaN checks before training.
- Unsupervised pattern discovery.
- Custom trading environment.
- PPO training with `stable-baselines3`.
- Commission and slippage support.
- R-multiple reward logic.
- Model saving.
- Backtest metrics.
- Equity curve plot.
- R-multiple distribution plot.
- Local acceptance tests.

## Tech Stack

- Python
- Google Colab
- pandas
- NumPy
- yfinance
- ta
- scikit-learn
- gymnasium
- stable-baselines3
- matplotlib

## RL Architecture

The notebook defines a custom trading environment.
The agent is trained with PPO.

The observation includes:

- recent return
- volume change
- RSI
- MACD histogram
- ATR
- rolling volatility
- discovered pattern id
- current position state

The action has three parts:

- trade decision
- stop-loss distance
- take-profit distance

The trade decision supports:

- hold
- enter long
- enter short
- close

Stop-loss and take-profit distances are based on ATR.

## Pattern Discovery

The notebook uses `MiniBatchKMeans`.

The model clusters standardized rolling windows of returns.
Each cluster becomes a market-pattern feature.

This is a simple and explainable baseline.
It keeps the notebook light enough for Colab.

## Reward System

Rewards are based on R-multiples.

```text
R-multiple = net trade return / initial trade risk
```

Initial trade risk is the distance from entry price to stop-loss.
Net trade return includes commission and slippage.

This rewards trades by profit relative to accepted risk.

## Backtest Outputs

The notebook reports:

- total return
- Sharpe ratio
- max drawdown
- number of trades
- win rate
- average R-multiple

The notebook also displays:

- equity curve
- R-multiple distribution

## How To Run In Colab

1. Open `Trading_RL_System.ipynb` in Google Colab.
2. Choose `Runtime > Restart and run all`.
3. Let the first cell install dependencies.
4. Confirm BTC-USD data downloads.
5. Confirm acceptance tests pass.
6. Review the final metrics table and plots.

## Local Tests

Run:

```bash
python3 acceptance_tests.py
```

Expected output:

```text
All acceptance tests passed.
```

## Assessment Sections Included

The notebook includes:

- How to run
- Stack and choices
- Final metrics table
- What broke / what surprised me
- Three weakest decisions
- What I would do with one more week

## Limitations

- K-Means is a simple pattern-discovery baseline.
- Hourly OHLCV bars cannot fully show intra-bar execution order.
- Position sizing is simplified.
- PPO training is intentionally short for assessment runtime.
- This is not live-trading advice.

## Future Improvements

- Add walk-forward validation.
- Compare PPO with baseline strategies.
- Tune hyperparameters with Optuna.
- Use a sequence-aware pattern model.
- Add realistic position sizing.
- Test lower-timeframe execution data.

## Submission Note

Run the notebook once in a fresh Colab runtime before submission.

Only report metrics produced by that run.
Do not invent or hardcode results.
