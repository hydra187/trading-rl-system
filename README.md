# 🚀 Trading RL System

> Reinforcement Learning based trading system using PPO, custom Gym environment, BTC-USD market data, feature engineering, backtesting, and risk-aware rewards.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![RL](https://img.shields.io/badge/Reinforcement-Learning-green)
![PPO](https://img.shields.io/badge/PPO-Agent-orange)
![StableBaselines3](https://img.shields.io/badge/StableBaselines3-RL-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Environment-purple)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# 📌 Project Overview

Trading RL System is a reinforcement learning-based trading simulator that trains a PPO (Proximal Policy Optimization) agent on BTC-USD market data and evaluates trading performance using custom rewards, backtesting metrics, and technical indicators.

This project demonstrates how reinforcement learning can be applied to financial market simulations through:

- Market feature engineering
- Reinforcement learning environments
- Risk-aware reward systems
- Pattern discovery
- Backtesting and evaluation

The project is built for learning and experimentation in AI, Reinforcement Learning, and quantitative trading.

---

# ✨ Features

✅ Historical BTC-USD data collection using `yfinance`  
✅ Technical indicator based feature engineering  
✅ Market pattern discovery using clustering  
✅ Custom Gymnasium trading environment  
✅ PPO reinforcement learning agent training  
✅ Long / Short / Hold / Close actions  
✅ ATR-based stop-loss and take-profit system  
✅ Commission + slippage handling  
✅ Risk-aware reward function (R-Multiple)  
✅ Backtesting and performance evaluation  
✅ Acceptance testing with synthetic data  
✅ Metrics and performance visualization  

---

# 🛠 Tech Stack

### Languages & Libraries

- Python
- NumPy
- pandas
- matplotlib
- scikit-learn
- yfinance
- gymnasium
- stable-baselines3

### Environment

- Google Colab
- Jupyter Notebook

---

# 🏗 Project Architecture

```txt
BTC-USD Market Data
          ↓
Feature Engineering
(RSI, ATR, MACD, Volatility)
          ↓
Pattern Discovery
(MiniBatchKMeans)
          ↓
Custom Trading Environment
(Gymnasium)
          ↓
PPO Agent Training
(Stable Baselines3)
          ↓
Trade Execution Simulation
(Long / Short / Close)
          ↓
Backtesting
          ↓
Performance Metrics + Graphs
```

---

# 📂 Repository Structure

```txt
trading-rl-system/
│
├── Trading_RL_System.ipynb
│   └── Main notebook for RL training and evaluation
│
├── acceptance_tests.py
│   └── Acceptance tests for environment logic
│
├── generate_notebook.py
│   └── Helper script
│
├── assets/
│   ├── equity_curve.png
│   ├── metrics.png
│   ├── training_output.png
│   └── acceptance_tests.png
│
└── README.md
```

---

# ⚙️ How It Works

The system follows an end-to-end RL trading pipeline.

### Step 1: Collect Market Data

Historical BTC-USD OHLCV data is downloaded using:

```python
yfinance
```

The dataset contains:

- Open
- High
- Low
- Close
- Volume

---

### Step 2: Feature Engineering

The system generates market features including:

- RSI (Relative Strength Index)
- ATR (Average True Range)
- MACD Histogram
- Volume Change
- Rolling Volatility
- Market Returns

These features are used as observations for the RL agent.

---

### Step 3: Market Pattern Discovery

The project uses:

```txt
MiniBatchKMeans
```

to identify simple market regimes/patterns.

Example:

- trending market
- sideways market
- volatile movement

The detected cluster ID becomes part of the observation space.

---

### Step 4: Custom Trading Environment

A custom Gymnasium environment simulates trading.

The environment supports:

### Observation Space

The model receives:

- RSI
- ATR
- MACD Histogram
- Volatility
- Returns
- Market pattern ID
- Position state

### Action Space

The PPO agent can:

| Action | Description |
|--------|-------------|
| 0 | Hold |
| 1 | Enter Long |
| 2 | Enter Short |
| 3 | Close Position |

---

### Step 5: Risk Management

Trades include:

- ATR-based stop-loss
- ATR-based take-profit
- commission cost
- slippage simulation

This improves realism during backtesting.

---

### Step 6: Reward System

Instead of rewarding raw profit, the project uses:

## R-Multiple Based Reward

```txt
R = Net Trade Return / Initial Trade Risk
```

This encourages the agent to optimize:

- better entries
- controlled risk
- better reward-to-risk ratio

rather than chasing random profits.

---

### Step 7: PPO Training

The system trains a:

```txt
PPO (Proximal Policy Optimization)
```

agent using:

```txt
stable-baselines3
```

The agent learns trading decisions through interaction with the environment.

---

### Step 8: Backtesting

After training, the system evaluates performance using:

- Total Return
- Sharpe Ratio
- Max Drawdown
- Number of Trades
- Win Rate
- Average R-Multiple

---

# 📊 Sample Outputs

Add screenshots inside:

```txt
assets/
```

Then insert them here.

## Equity Curve

```md
![Equity Curve](assets/equity_curve.png)
```

## Training Output

```md
![Training Output](assets/training_output.png)
```

## Metrics

```md
![Metrics](assets/metrics.png)
```

## Acceptance Tests

```md
![Acceptance Tests](assets/acceptance_tests.png)
```

---

# ▶️ How To Run

## Run in Google Colab

1. Open `Trading_RL_System.ipynb`
2. Click:

```txt
Runtime → Run all
```

3. Wait for dependencies to install
4. Review metrics and generated graphs

---

## Run Locally

Install dependencies:

```bash
pip install pandas numpy matplotlib gymnasium scikit-learn yfinance stable-baselines3
```

Run notebook.

---

## Run Acceptance Tests

```bash
python3 acceptance_tests.py
```

Expected output:

```txt
All acceptance tests passed.
```

---

# 📈 Example Performance Metrics

Example metrics generated during evaluation:

| Metric | Description |
|--------|-------------|
| Total Return | Final return percentage |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Worst portfolio drop |
| Win Rate | Successful trades |
| Avg R-Multiple | Reward per risk |

---

# 🚧 Current Limitations

- Limited PPO training duration for faster execution
- Simplified trading simulation
- Uses historical OHLCV data only
- No live market execution
- Position sizing is simplified

---

# 🔮 Future Improvements

- Streamlit trading dashboard
- Live market paper trading
- Hyperparameter tuning
- Multi-asset trading
- Walk-forward validation
- Model comparison
- WebSocket real-time price updates
- Better portfolio management
- Advanced risk management

---

# 🧪 Acceptance Testing

The repository includes:

```txt
acceptance_tests.py
```

to verify:

- environment logic
- trade execution
- reward calculation
- position handling

using synthetic market data.

---

# 🎯 Learning Outcomes

This project helped explore:

- Reinforcement Learning
- PPO Algorithms
- Custom RL environments
- Feature engineering
- Quantitative trading basics
- Backtesting systems
- Risk-aware reward engineering
- Financial data analysis

---

# ⚠️ Disclaimer

This project is for educational and research purposes only.

It is **NOT financial advice** and should not be used for live trading without proper testing, validation, and risk management.

---

