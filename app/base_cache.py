"""基座乘法结果缓存。

缓存键是**补零后的 float64 向量的原始字节**（``ndarray.tobytes()``），
因此只有字节完全一致的输入才会命中同一项。

本模块只维护纯数据结构，不做加锁；调用方（``LoRAScheduler``）必须
持有同一个锁后再调用这些方法，保证状态迁移与条件变量通知是原子的。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

# 条目状态
COMPUTING = "computing"
READY = "ready"

# 基座缓存的硬上限：最多 8 项（计算中的条目也占名额）
MAX_CAPACITY = 8


@dataclass
class BaseEntry:
    """一笔基座乘法缓存的条目。"""

    key: bytes
    state: str = COMPUTING
    value: np.ndarray | None = None
    owner_id: int | None = None       # 实际执行 W0·x 的请求 id
    waiters: list = field(default_factory=list)  # 正在等待本次计算的请求记录
    users: int = 0                    # 已拿到结果、尚未结束推理的请求数
    seq: int = 0                      # 插入顺序，LRU 平局时使用
    created_at: float = field(default_factory=time.time)
    ready_at: float = 0.0
    last_used_at: float = field(default_factory=time.time)

    @property
    def key_label(self) -> str:
        """给网页展示的短标识（不暴露完整向量）。"""
        return hashlib.sha256(self.key).hexdigest()[:12]

    @property
    def evictable(self) -> bool:
        """只有算完且无人等待、无人使用的条目才可淘汰。"""
        return self.state == READY and not self.waiters and self.users == 0

    def as_dict(self) -> dict:
        return {
            "key": self.key_label,
            "state": self.state,
            "owner_id": self.owner_id,
            "waiters": len(self.waiters),
            "users": self.users,
            "created_at": self.created_at,
            "ready_at": self.ready_at or None,
            "last_used_at": self.last_used_at,
        }


class BaseResultCache:
    """最多 8 项的基座结果缓存（计算中的条目同样占名额）。"""

    def __init__(self, capacity: int = MAX_CAPACITY):
        # 配置只能调小，不能突破 8 项硬上限
        self.capacity = max(1, min(int(capacity), MAX_CAPACITY))
        self.entries: dict[bytes, BaseEntry] = {}
        self._seq = 0
        # 统计
        self.hits = 0          # 命中已算完的条目
        self.joins = 0         # 命中正在计算的条目并共用结果
        self.misses = 0        # 新建条目并实际计算
        self.bypasses = 0      # 缓存满且无法淘汰，绕过缓存计算
        self.evictions = 0
        self.failures = 0

    # ------------------------------------------------------------------ #
    # 查询 / 容量
    # ------------------------------------------------------------------ #
    def get(self, key: bytes) -> BaseEntry | None:
        return self.entries.get(key)

    @property
    def size(self) -> int:
        return len(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def make_room(self) -> bool:
        """为新条目腾一个名额。

        仅淘汰 ``evictable`` 的条目（按最近使用时间 LRU）。缓存未满返回
        True 但不淘汰。所有条目都在计算/被使用时返回 False。
        """
        if len(self.entries) < self.capacity:
            return True
        candidates = [e for e in self.entries.values() if e.evictable]
        if not candidates:
            return False
        victim = min(candidates, key=lambda e: (e.last_used_at, e.seq))
        del self.entries[victim.key]
        self.evictions += 1
        return True

    # ------------------------------------------------------------------ #
    # 状态迁移
    # ------------------------------------------------------------------ #
    def insert_computing(self, key: bytes, owner_id: int) -> BaseEntry:
        """占位并返回新条目；调用前须确保 ``make_room`` 成功。"""
        self._seq += 1
        entry = BaseEntry(
            key=key, state=COMPUTING, owner_id=owner_id, seq=self._seq,
        )
        entry.users = 1  # 发起者自己即将使用结果
        self.entries[key] = entry
        self.misses += 1
        return entry

    def add_waiter(self, entry: BaseEntry, record) -> None:
        entry.waiters.append(record)
        entry.last_used_at = time.time()
        self.joins += 1

    def add_user(self, entry: BaseEntry) -> None:
        entry.users += 1
        entry.last_used_at = time.time()

    def release_user(self, key: bytes) -> None:
        entry = self.entries.get(key)
        if entry is not None and entry.users > 0:
            entry.users -= 1
            entry.last_used_at = time.time()

    def finish(self, entry: BaseEntry, value: np.ndarray) -> list:
        """标记计算成功，把等待者提升为使用者并返回它们。"""
        entry.value = value
        entry.state = READY
        entry.ready_at = time.time()
        entry.last_used_at = entry.ready_at
        waiters = list(entry.waiters)
        entry.waiters = []
        for _ in waiters:
            entry.users += 1
        return waiters

    def fail(self, entry: BaseEntry) -> list:
        """计算失败：移出缓存并返回全部等待者；相同输入下次会重算。"""
        self.failures += 1
        waiters = list(entry.waiters)
        entry.waiters = []
        if self.entries.get(entry.key) is entry:
            del self.entries[entry.key]
        entry.value = None
        return waiters

    def stats(self) -> dict:
        entries = [e.as_dict() for e in self.entries.values()]
        entries.sort(key=lambda e: e["created_at"])
        return {
            "capacity": self.capacity,
            "size": len(self.entries),
            "computing": sum(1 for e in self.entries.values() if e.state == COMPUTING),
            "ready": sum(1 for e in self.entries.values() if e.state == READY),
            "waiting": sum(len(e.waiters) for e in self.entries.values()),
            "in_use": sum(e.users for e in self.entries.values()),
            "hits": self.hits,
            "joins": self.joins,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "failures": self.failures,
            "entries": entries,
        }
