"""同一个线性模型的 NumPy 实现 + 三种大小不同的 LoRA 适配器。

模型::

    y = W0 @ x + (alpha / r) * B @ (A @ x)

W0 是所有请求共享的基座权重（常驻，不计入适配器内存额度）；
每个适配器包含矩阵 A(rank, d_in) 与 B(d_out, rank)，其字节数
``(d_in * rank + rank * d_out) * float64字节数`` 就是它在内存额度里
按完整大小占用的空间（加载中也一样）。
"""
from __future__ import annotations

import hashlib

import numpy as np

# 三种大小不同的适配器：小 / 中 / 大
ADAPTER_SPECS = [
    {"id": "lora-small", "rank": 8, "alpha": 16.0},
    {"id": "lora-medium", "rank": 64, "alpha": 16.0},
    {"id": "lora-large", "rank": 256, "alpha": 16.0},
]

_DTYPE = np.float64
_DTYPE_BYTES = np.dtype(_DTYPE).itemsize


def _stable_rng(*parts: object) -> np.random.Generator:
    """由名称生成确定性随机源，保证每次启动权重一致。"""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


class LinearLoRAModel:
    """带 LoRA 的单层线性模型，纯 NumPy 计算。"""

    def __init__(self, dim_in: int = 1024, dim_out: int = 1024):
        self.dim_in = dim_in
        self.dim_out = dim_out
        rng = _stable_rng("base", dim_in, dim_out)
        # 基座权重刻意做小，保证演示输出稳定可读
        self.W0 = rng.normal(0.0, 0.02, size=(dim_out, dim_in)).astype(_DTYPE)

    @property
    def base_nbytes(self) -> int:
        return int(self.W0.nbytes)

    def adapter_nbytes(self, rank: int) -> int:
        """单个适配器完整驻留时的字节数（A 与 B 两块权重之和）。"""
        return int((self.dim_in * rank + rank * self.dim_out) * _DTYPE_BYTES)

    def build_adapter(self, adapter_id: str, rank: int) -> dict[str, np.ndarray]:
        """构造适配器权重 A / B（模拟从磁盘加载到内存）。"""
        rng_a = _stable_rng(adapter_id, rank, "A")
        rng_b = _stable_rng(adapter_id, rank, "B")
        a = rng_a.normal(0.0, 0.05, size=(rank, self.dim_in)).astype(_DTYPE)
        b = rng_b.normal(0.0, 0.05, size=(self.dim_out, rank)).astype(_DTYPE)
        return {"A": a, "B": b}

    def forward(self, x: np.ndarray, adapter: dict | None, rank: int, alpha: float) -> np.ndarray:
        """线性前向；adapter 为 None 时只走基座。"""
        base = self.W0 @ x
        if adapter is None:
            return base
        scaling = alpha / rank
        delta = scaling * (adapter["B"] @ (adapter["A"] @ x))
        return base + delta
