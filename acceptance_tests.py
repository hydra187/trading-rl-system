"""Acceptance tests for the Trading RL System environment logic.

These tests use synthetic prices.
They do not download market data.
They do not train a model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeInfo:
    trade_closed: bool = False
    pnl: float = 0.0
    r_multiple: float = 0.0


class TradingEnv:
    """Small test environment matching the notebook's core trade logic."""

    def __init__(self, data, commission=0.0005, slippage=0.0005):
        self.data = data.reset_index(drop=True).copy()
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.friction = self.commission + self.slippage
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.initial_stop_price = 0.0
        self.stop_price = 0.0
        self.take_profit_price = 0.0

    def step(self, action):
        decision, stop_mult, target_mult = self._decode_action(action)
        row = self.data.loc[self.current_step]

        price = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        atr = float(row["atr"])

        reward = 0.0
        info = TradeInfo()

        if self.position != 0:
            hit_stop = self._hit_stop(low=low, high=high)
            hit_target = self._hit_target(low=low, high=high)

            if hit_stop or hit_target:
                exit_price = self.stop_price if hit_stop else self.take_profit_price
                reward, info = self._close_position(exit_price)

        if decision == "close" and self.position != 0:
            close_reward, close_info = self._close_position(price)
            reward += close_reward
            info = close_info

        target_side = self._target_side(decision)

        if target_side != 0 and self.position != target_side:
            reward += self._open_position(
                side=target_side,
                price=price,
                atr=atr,
                stop_mult=stop_mult,
                target_mult=target_mult,
            )

        self.current_step += 1
        return reward, info

    def _decode_action(self, action):
        action = np.asarray(action, dtype=float)
        decision_value = float(np.clip(action[0], -1.0, 1.0))

        if decision_value >= 0.50:
            decision = "enter_long"
        elif decision_value <= -0.50:
            decision = "enter_short"
        elif -0.20 <= decision_value <= 0.20:
            decision = "hold"
        else:
            decision = "close"

        stop_mult = self._scale_to_range(action[1], low=1.0, high=5.0)
        target_mult = self._scale_to_range(action[2], low=1.0, high=5.0)

        return decision, stop_mult, target_mult

    @staticmethod
    def _scale_to_range(value, low, high):
        clipped = float(np.clip(value, -1.0, 1.0))
        fraction = (clipped + 1.0) / 2.0
        return low + fraction * (high - low)

    @staticmethod
    def _target_side(decision):
        if decision == "enter_long":
            return 1
        if decision == "enter_short":
            return -1
        return 0

    def _hit_stop(self, low, high):
        if self.position == 1:
            return low <= self.stop_price
        if self.position == -1:
            return high >= self.stop_price
        return False

    def _hit_target(self, low, high):
        if self.position == 1:
            return high >= self.take_profit_price
        if self.position == -1:
            return low <= self.take_profit_price
        return False

    def _risk_fraction(self):
        if self.entry_price <= 0:
            return 0.0

        risk = abs(self.entry_price - self.initial_stop_price)
        return risk / self.entry_price

    def _open_position(self, side, price, atr, stop_mult, target_mult):
        self.position = int(side)
        self.entry_price = float(price)

        risk_distance = max(float(atr) * float(stop_mult), 1e-8)

        if side == 1:
            self.initial_stop_price = self.entry_price - risk_distance
            self.take_profit_price = self.entry_price + risk_distance * target_mult
        else:
            self.initial_stop_price = self.entry_price + risk_distance
            self.take_profit_price = self.entry_price - risk_distance * target_mult

        self.stop_price = self.initial_stop_price

        risk_fraction = risk_distance / self.entry_price
        return -(self.friction / risk_fraction)

    def _close_position(self, exit_price):
        exit_price = float(exit_price)

        if self.position == 1:
            gross_return = (exit_price - self.entry_price) / self.entry_price
        else:
            gross_return = (self.entry_price - exit_price) / self.entry_price

        net_return = gross_return - self.friction
        risk_fraction = self._risk_fraction()

        if risk_fraction > 0:
            r_multiple = net_return / risk_fraction
        else:
            r_multiple = 0.0

        self.position = 0
        self.entry_price = 0.0
        self.initial_stop_price = 0.0
        self.stop_price = 0.0
        self.take_profit_price = 0.0

        info = TradeInfo(
            trade_closed=True,
            pnl=net_return,
            r_multiple=r_multiple,
        )

        return r_multiple, info


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
    total_reward = 0.0

    for _ in range(10):
        reward, _ = env.step([0.0, -1.0, -1.0])
        total_reward += reward

    assert abs(total_reward) < 1e-12


def test_open_then_close_loses_friction_cost():
    env = TradingEnv(make_acceptance_data())

    open_reward, _ = env.step([1.0, -1.0, -1.0])
    close_reward, info = env.step([0.3, -1.0, -1.0])

    assert info.trade_closed
    assert open_reward + close_reward < 0.0


def test_three_r_take_profit_reward():
    env = TradingEnv(
        make_acceptance_data(),
        commission=0.0,
        slippage=0.0,
    )

    env.position = 1
    env.entry_price = 100.0
    env.initial_stop_price = 99.0
    env.stop_price = 99.0
    env.take_profit_price = 103.0

    env.data.loc[env.current_step, "High"] = 103.25

    reward, info = env.step([0.0, -1.0, -1.0])

    assert info.trade_closed
    assert np.isclose(info.r_multiple, 3.0, atol=0.05)
    assert np.isclose(reward, 3.0, atol=0.05)


def run_tests():
    test_do_nothing_policy_reward_is_zero()
    test_open_then_close_loses_friction_cost()
    test_three_r_take_profit_reward()
    print("All acceptance tests passed.")


if __name__ == "__main__":
    run_tests()
