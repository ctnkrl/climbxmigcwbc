# Copyright (c) 2026 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np


class CmdFilter:
    """命令滤波器：限幅 → 速率限制 → 指数平滑"""

    def __init__(self, low: float | None, high: float | None, rate_limit: float | None, alpha: float | None):
        """
        Args:
            low, high: clip_range
            rate_limit: 每秒的变化率
            alpha: 指数滤波系数, 1.0不滤波, 0.0全等于last
        """
        self.low = low
        self.high = high
        self.rate_limit = rate_limit
        if alpha is not None:
            assert (alpha >= 0.0) and (alpha <= 1.0)
        else:
            alpha = 1.0
        self.alpha = alpha
        self.last_value = None

    def filter(self, value: np.ndarray, dt: float) -> np.ndarray:
        """对输入值做三级串行滤波"""
        # 1. 限幅
        if (self.low is None) and (self.high is None):
            pass
        else:
            value = np.clip(value, self.low, self.high)  # 允许low/high其中一个为None

        if self.last_value is None:
            # 第一次只clip
            self.last_value = value
            return value

        # 2. 变化率限制
        if self.rate_limit:
            max_delta = self.rate_limit * dt
            delta = np.clip(value - self.last_value, -max_delta, max_delta)
            limited = self.last_value + delta
        else:
            limited = value

        # 3. 指数平滑
        smoothed = (1.0 - self.alpha) * self.last_value + self.alpha * limited

        self.last_value = smoothed
        return self.last_value

    def reset(self, value: np.ndarray = None):
        """重置内部状态"""
        self.last_value = None if value is None else value
