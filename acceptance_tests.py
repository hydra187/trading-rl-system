import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    def __init__(self, df, commission=0.0005, slippage=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.friction = self.commission + self.slippage
        self.feature_cols = [
            "returns",
            "volume_change",
            "rsi",
            "macd",
            "atr",
            "volatility",
            "pattern",
        ]
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
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.df.loc[self.current_step, self.feature_cols].astype(float).values
        obs = np.append(obs, float(self.position))
        return np.nan_to_num(obs).astype(np.float32)

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

        sl_mult = 1.0 + (
            (float(np.clip(action[1], -1.0, 1.0)) + 1.0) / 2.0
        ) * 4.0
        tp_mult = 1.0 + (
            (float(np.clip(action[2], -1.0, 1.0)) + 1.0) / 2.0
        ) * 4.0
        return decision, sl_mult, tp_mult

    def _risk_fraction(self):
        if self.entry_price <= 0:
            return 0.0
        return abs(self.entry_price - self.initial_stop_price) / self.entry_price

    def _close_position(self, exit_price):
        if self.position == 1:
            gross_return = (exit_price - self.entry_price) / self.entry_price
        else:
            gross_return = (self.entry_price - exit_price) / self.entry_price

        net_return = gross_return - self.friction
        risk_fraction = self._risk_fraction()
        r_multiple = net_return / risk_fraction if risk_fraction > 0 else 0.0
        self.position = 0
        return net_return, r_multiple

    def _open_position(self, side, price, atr, sl_mult, tp_mult):
        self.position = int(side)
        self.entry_price = float(price)
        risk_distance = max(float(atr) * float(sl_mult), 1e-8)
        if side == 1:
            self.initial_stop_price = self.entry_price - risk_distance
            self.take_profit_price = self.entry_price + risk_distance * tp_mult
        else:
            self.initial_stop_price = self.entry_price + risk_distance
            self.take_profit_price = self.entry_price - risk_distance * tp_mult
        self.stop_price = self.initial_stop_price
        return -(self.friction / (risk_distance / self.entry_price))

    def step(self, action):
        reward = 0.0
        info = {"trade_closed": False, "pnl": 0.0, "r_multiple": 0.0}
        row = self.df.loc[self.current_step]
        price = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        decision, sl_mult, tp_mult = self._decode_action(action)

        if self.position != 0:
            hit_stop = (
                self.position == 1
                and low <= self.stop_price
                or self.position == -1
                and high >= self.stop_price
            )
            hit_take_profit = (
                self.position == 1
                and high >= self.take_profit_price
                or self.position == -1
                and low <= self.take_profit_price
            )
            if hit_stop or hit_take_profit:
                exit_price = self.stop_price if hit_stop else self.take_profit_price
                pnl, r_multiple = self._close_position(exit_price)
                reward += r_multiple
                info.update(
                    {"trade_closed": True, "pnl": pnl, "r_multiple": r_multiple}
                )

        if decision == "close" and self.position != 0:
            pnl, r_multiple = self._close_position(price)
            reward += r_multiple
            info.update({"trade_closed": True, "pnl": pnl, "r_multiple": r_multiple})

        target_side = 1 if decision == "enter_long" else -1 if decision == "enter_short" else 0
        if target_side != 0 and self.position != target_side:
            reward += self._open_position(
                target_side,
                price,
                float(row["atr"]),
                sl_mult,
                tp_mult,
            )

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), terminated, False, info


def make_acceptance_data():
    close = np.full(20, 100.0)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(20, 1000.0),
            "returns": np.zeros(20),
            "volume_change": np.zeros(20),
            "rsi": np.full(20, 50.0),
            "macd": np.zeros(20),
            "atr": np.full(20, 1.0),
            "volatility": np.full(20, 0.01),
            "pattern": np.zeros(20, dtype=int),
        }
    )


def test_do_nothing_policy_reward_is_zero():
    env = TradingEnv(make_acceptance_data())
    env.reset()
    total_reward = 0.0
    for _ in range(10):
        _, reward, _, _, _ = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))
        total_reward += reward
    assert abs(total_reward) < 1e-12


def test_open_then_close_loses_friction_cost():
    env = TradingEnv(make_acceptance_data())
    env.reset()
    _, open_reward, _, _, _ = env.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    _, close_reward, _, _, info = env.step(np.array([0.3, -1.0, -1.0], dtype=np.float32))
    assert info["trade_closed"]
    assert open_reward + close_reward < 0


def test_three_r_take_profit_reward():
    env = TradingEnv(make_acceptance_data(), commission=0.0, slippage=0.0)
    env.reset()
    env.position = 1
    env.entry_price = 100.0
    env.initial_stop_price = 99.0
    env.stop_price = 99.0
    env.take_profit_price = 103.0
    env.df.loc[env.current_step, "High"] = 103.25

    _, reward, _, _, info = env.step(np.array([0.0, -1.0, -1.0], dtype=np.float32))
    assert info["trade_closed"]
    assert np.isclose(info["r_multiple"], 3.0, atol=0.05)
    assert np.isclose(reward, 3.0, atol=0.05)


if __name__ == "__main__":
    test_do_nothing_policy_reward_is_zero()
    test_open_then_close_loses_friction_cost()
    test_three_r_take_profit_reward()
    print("All acceptance tests passed.")
