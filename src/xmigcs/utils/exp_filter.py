# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np
import math

from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

class expFilter:
    def __init__(self, alpha=None, tau=None):
        """
        指数滤波器\\
        alpha=0,完全不滤波,等于current\\
        alpha=1,全部等于last

        半衰期tau和alpha的关系:
        $\alpha^{\tau}=0.5$

        $\tau = \frac{\ln(0.5)}{\ln(\alpha)}$
        """
        if (alpha is None) and (tau is None):
            raise Exception("缺少alpha或tau参数")
        if (alpha is not None) and (tau is not None):
            raise Exception("alpha和tau参数重复")
        if alpha is None:
            alpha = math.exp(math.log(0.5) / tau)
        assert (alpha > 0.0) and (alpha < 1.0)
        if tau is None:
            tau = math.log(0.5) / math.log(alpha)
        self.alpha = alpha
        self.tau = tau
        self.last = None

    def filter(self, current):
        if self.last is None:
            if type(current) is np.ndarray:
                self.last = np.zeros_like(current)
            elif type(current) in [int, float, np.float32, np.float64]:
                self.last = 0.0
            else:
                logger.error(f"不支持的滤波类型: {type(current)}")
                raise NotImplementedError
        filtered = current * (1 - self.alpha) + self.last * self.alpha
        self.last = filtered
        return filtered

    def reset(self):
        self.last = None
