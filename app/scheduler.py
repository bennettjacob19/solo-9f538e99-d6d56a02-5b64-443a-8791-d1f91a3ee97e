"""适配器内存调度器 + 基座乘法结果缓存（本项目核心）。

基座复用
--------
模型输出 ``y = W0·x + Δ_adapter(x)``。评测时同一个（补零后的 float64）
向量常被拿去分别跑几款适配器，因此把 ``W0·x`` 按向量原始字节缓存：

* 缓存 key 是补零后 ``float64`` 向量的 ``tobytes()``，字节完全一致才算同一项。
* 容量固定 8 项，**计算中的条目也占名额**。
* 命中「正在计算」的条目：请求登记为等待者并**交还推理线程**；基座算完后
  重新排到线程池，只计算自己的适配器增量。
* 命中「已算完」的条目：直接复用，不产生新的基座乘法。
* 只有已算完、且无人等待、无人使用的条目才能被 LRU 淘汰。
* 缓存满且无法淘汰时，新向量绕过缓存照常在本请求线程里计算。
* 基座计算失败：删除条目，发起者与全部等待者都报错；失败不缓存，
  下一笔相同输入的请求会重新计算。
* ``base_multiplies`` 统计实际执行 ``W0 @ x`` 的次数（含绕过缓存与重试）。

等待共享基座结果时不占推理线程，但请求对适配器权重的认领（``in_use``）
仍然保留，因此适配器不会被错误卸载，适配器内存额度核算与原来一致。

内存规则
--------
* 启动时给定适配器权重内存上限（字节），基座模型不计入。
* 适配器一旦**开始加载**，就按完整大小计入 ``committed``，加载过程中
  其他适配器可用额度相应减少。
* 已驻留适配器的请求直接推理；同一适配器的并发请求只会触发一次加载，
  其余请求等待同一批加载（同一个 generation）完成。
* 用完的权重不主动卸载；只有新的加载装不下时，才按 LRU 顺序卸载
  处于闲置（``in_use == 0``）的已驻留适配器。
* 暂时装不下的请求挂起等待，直到有额度腾出再继续。
* 单个适配器超过内存上限：请求直接拒绝，不占用任何额度。
* 加载失败：本次加载的所有等待请求都收到错误，占用额度立即释放，
  后续请求会重新发起加载。
* 慢加载期间：已驻留的适配器继续接单；放得下的冷适配器也可以并行
  开始加载（多块适配器允许同时在途加载）。
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .base_cache import BaseResultCache, COMPUTING
from .model import LinearLoRAModel

# ---- 请求状态 ----
PENDING = "pending"        # 已提交，等待内存/加载
LOADING = "loading"        # 适配器加载中（请求在等权重）
BASE_WAIT = "base_wait"    # 等待共享的基座乘法（已交还推理线程）
RUNNING = "running"        # 权重就绪，推理中
SUCCESS = "success"
FAILED = "failed"
REJECTED = "rejected"

ALL_REQUEST_STATES = (PENDING, LOADING, BASE_WAIT, RUNNING,
                      SUCCESS, FAILED, REJECTED)
TERMINAL_STATES = {SUCCESS, FAILED, REJECTED}

# 基座结果来源（写入请求记录，网页展示）
BASE_COMPUTED = "computed"  # 未命中，本次实际执行 W0·x
BASE_CACHE = "cache"        # 命中已算完的条目
BASE_JOINED = "joined"      # 命中计算中的条目，共用本次计算
BASE_BYPASS = "bypass"      # 缓存满且无法淘汰，绕过缓存

# ---- 适配器状态 ----
A_UNLOADED = "unloaded"
A_LOADING = "loading"
A_RESIDENT = "resident"
A_FAILED = "failed"


@dataclass
class Adapter:
    id: str
    rank: int
    alpha: float
    nbytes: int
    state: str = A_UNLOADED
    weights: dict | None = None
    in_use: int = 0                 # 正在服务的请求数
    gen: int = 0                    # 当前加载批次编号
    load_attempts: int = 0          # 实际开始加载的次数
    load_successes: int = 0         # 加载成功次数
    last_used_at: float = 0.0       # LRU 依据（最近一次释放时刻）
    failed_gens: set = field(default_factory=set)  # 已失败的批次，唤醒等待者用


@dataclass
class RequestRecord:
    id: int
    adapter_id: str
    x: object
    status: str = PENDING
    detail: str = "排队中"
    result: list | None = None
    result_norm: float | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    wait_ms: float | None = None
    infer_ms: float | None = None
    loaded_for: bool = False        # 本次请求是否触发了实际加载
    claimed: bool = False           # 是否已认领（计入适配器 in_use）
    gen: int | None = None
    # ---- 基座复用 ----
    base_key: bytes | None = None       # 补零后 float64 向量的原始字节
    base_value: np.ndarray | None = None
    base_source: str | None = None      # computed / cache / joined / bypass
    base_cached: bool = False           # 是否持有缓存条目使用权
    events: list = field(default_factory=list)

    def add_event(self, message: str) -> None:
        self.events.append({"at": round(time.time(), 3), "message": message})


class LoRAScheduler:
    def __init__(
        self,
        model: LinearLoRAModel,
        adapter_metas: list[dict],
        memory_limit: int,
        load_delay: float = 2.0,
        infer_delay: float = 1.0,
        fail_next: list[str] | None = None,
        max_workers: int = 64,
        history: int = 200,
        base_cache_capacity: int = 8,
        base_delay: float = 0.0,
    ):
        self.model = model
        self.memory_limit = int(memory_limit)
        self.load_delay = float(load_delay)
        self.infer_delay = float(infer_delay)
        self.fail_next: set[str] = set(fail_next or [])
        self.max_workers = max_workers

        self.adapters: dict[str, Adapter] = {
            meta["id"]: Adapter(
                id=meta["id"], rank=meta["rank"], alpha=meta["alpha"],
                nbytes=int(meta["nbytes"]),
            )
            for meta in adapter_metas
        }

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lora-req"
        )
        self._req_seq = itertools.count(1)
        self._requests: deque[RequestRecord] = deque(maxlen=history)
        self.committed = 0             # 已占用额度（含加载中）
        self.evictions = 0
        self.current_loading: set[str] = set()  # 当前在途加载的适配器 id
        self.base_cache = BaseResultCache(capacity=base_cache_capacity)
        self.base_multiplies = 0       # 实际执行 W0·x 的次数
        self.base_delay = max(0.0, float(base_delay))  # 可选的基座乘法模拟耗时
        # 测试/演示用：这些基座输入字节的下一次计算会失败（一次性）
        self.fail_next_base: set[bytes] = set()
        # 测试用：注册在此的 threading.Event 会在 teardown 时强制 set，
        # 避免阻塞在门闩上的 loader 线程跨测试泄漏
        self.test_gates: list = []

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def submit(self, adapter_id: str, x) -> RequestRecord:
        """提交一次推理请求，立即返回记录；实际处理在线程池中进行。

        输入在这里统一补零成 ``dim_in`` 维 ``float64`` 向量，缓存键就是
        该规范向量的原始字节，因此不同长度但补零后相同的输入可复用。
        """
        canonical = np.zeros(self.model.dim_in, dtype=np.float64)
        given = np.asarray(x, dtype=np.float64).ravel()[: self.model.dim_in]
        canonical[: given.size] = given
        with self._lock:
            record = RequestRecord(
                id=next(self._req_seq), adapter_id=adapter_id, x=canonical,
                base_key=canonical.tobytes(),
            )
            record.add_event("请求已提交")
            self._requests.appendleft(record)

            adapter = self.adapters.get(adapter_id)
            if adapter is None:
                self._mark_terminal(record, REJECTED, f"未知适配器 {adapter_id}")
                return record
            if adapter.nbytes > self.memory_limit:
                # 单个适配器超限：直接拒绝，不占额度
                self._mark_terminal(
                    record, REJECTED,
                    f"适配器大小 {self._fmt(adapter.nbytes)} 超过内存上限 "
                    f"{self._fmt(self.memory_limit)}",
                )
                return record

        self._pool.submit(self._process, record)
        return record

    def update_config(self, *, load_delay=None, infer_delay=None,
                      base_delay=None) -> dict:
        with self._lock:
            if load_delay is not None:
                self.load_delay = max(0.0, float(load_delay))
            if infer_delay is not None:
                self.infer_delay = max(0.0, float(infer_delay))
            if base_delay is not None:
                self.base_delay = max(0.0, float(base_delay))
            return {
                "load_delay": self.load_delay,
                "infer_delay": self.infer_delay,
                "base_delay": self.base_delay,
            }

    def set_fail_next(self, adapter_id: str, value: bool) -> Adapter:
        with self._lock:
            adapter = self.adapters.get(adapter_id)
            if adapter is None:
                raise KeyError(adapter_id)
            if value:
                self.fail_next.add(adapter_id)
            else:
                self.fail_next.discard(adapter_id)
            return adapter

    def snapshot(self) -> dict:
        """供网页轮询的整体状态快照。"""
        with self._lock:
            adapters = []
            for a in self.adapters.values():
                adapters.append({
                    "id": a.id,
                    "rank": a.rank,
                    "nbytes": a.nbytes,
                    "state": a.state,
                    "in_use": a.in_use,
                    "load_attempts": a.load_attempts,
                    "load_successes": a.load_successes,
                    "last_used_at": a.last_used_at,
                    "will_fail_next": a.id in self.fail_next,
                })
            requests = [
                {
                    "id": r.id,
                    "adapter_id": r.adapter_id,
                    "status": r.status,
                    "detail": r.detail,
                    "error": r.error,
                    "result": r.result,
                    "result_norm": r.result_norm,
                    "wait_ms": r.wait_ms,
                    "infer_ms": r.infer_ms,
                    "loaded_for": r.loaded_for,
                    "base_source": r.base_source,
                    "base_cached": r.base_cached,
                    "events": list(r.events),
                }
                for r in self._requests
            ]
            counts = {s: 0 for s in ALL_REQUEST_STATES}
            for r in self._requests:
                counts[r.status] += 1
            return {
                "memory": {
                    "limit": self.memory_limit,
                    "committed": self.committed,
                    "free": self.memory_limit - self.committed,
                    "base_model_nbytes": self.model.base_nbytes,
                    "evictions": self.evictions,
                },
                "base": {
                    "multiplies": self.base_multiplies,
                    "delay": self.base_delay,
                    **self.base_cache.stats(),
                },
                "config": {
                    "load_delay": self.load_delay,
                    "infer_delay": self.infer_delay,
                    "base_delay": self.base_delay,
                    "max_workers": self.max_workers,
                    "dim_in": self.model.dim_in,
                    "dim_out": self.model.dim_out,
                },
                "adapters": adapters,
                "requests": requests,
                "counts": counts,
            }

    def clear_requests(self) -> None:
        with self._lock:
            active = [r for r in self._requests if r.status not in TERMINAL_STATES]
            self._requests = deque(active, maxlen=self._requests.maxlen)

    def shutdown(self) -> None:
        self.force_release_test_gates()
        # 还在等共享基座的请求不会再有结果（进程正在退出），明确标记失败
        with self._lock:
            for record in self._requests:
                if record.status == BASE_WAIT:
                    adapter = self.adapters.get(record.adapter_id)
                    if adapter is not None:
                        self._release_claim(adapter, record)
                    record.base_cached = False
                    self._mark_terminal(record, FAILED, "调度器关闭，未取得基座结果")
        self._pool.shutdown(wait=False)

    def force_release_test_gates(self) -> None:
        """测试辅助：解除所有已注册门闩，放行被挂起的加载/推理线程。"""
        for gate in list(self.test_gates):
            gate.set()

    # ------------------------------------------------------------------ #
    # 请求处理主循环（工作线程内运行）
    # ------------------------------------------------------------------ #
    def _process(self, record: RequestRecord) -> None:
        while True:
            with self._lock:
                adapter = self.adapters[record.adapter_id]

                # 唤醒后复查：本请求等待的那一批加载已经失败
                if record.gen is not None and record.gen in adapter.failed_gens:
                    self._release_claim(adapter, record)
                    self._mark_terminal(
                        record, FAILED,
                        f"适配器 {adapter.id} 加载失败（第 {record.gen} 批）",
                    )
                    return

                if adapter.state == A_RESIDENT:
                    # 已驻留：慢加载期间也照常接单。
                    # 若是从加载等待转来的，之前已认领；新请求在此认领一次。
                    self._claim(adapter, record)
                    adapter.last_used_at = time.time()
                    record.status = RUNNING
                    record.detail = "适配器已驻留，直接推理"
                    record.add_event("命中已驻留权重，开始推理")
                    if record.started_at is None:
                        record.started_at = time.time()
                    weights = adapter.weights
                    break

                if adapter.state == A_LOADING:
                    # 同适配器并发请求：复用同一次加载，只等不重复加载。
                    # 加入等待即认领，保证加载完成前 in_use>=1，绝不被误卸载。
                    self._claim(adapter, record)
                    record.status = LOADING
                    record.gen = adapter.gen
                    record.detail = f"等待适配器加载（第 {adapter.gen} 批，复用并发加载）"
                    record.add_event(record.detail)
                    self._cond.wait(timeout=30)
                    continue

                # adapter.state in (UNLOADED, FAILED)：尝试发起一次新加载
                if not self._try_start_load_locked(adapter, record):
                    record.status = PENDING
                    record.detail = (
                        f"等待内存腾出（需要 {self._fmt(adapter.nbytes)}，"
                        f"空闲 {self._fmt(self.memory_limit - self.committed)}）"
                    )
                    if not record.events or record.events[-1]["message"] != record.detail:
                        record.add_event(record.detail)
                    self._cond.wait(timeout=30)
                    continue

                # 已开始加载：发起者立即认领，在加载全程占住 in_use
                self._claim(adapter, record)
                record.status = LOADING
                record.loaded_for = True
                record.gen = adapter.gen
                if record.started_at is None:
                    record.started_at = time.time()
                record.add_event(f"触发适配器加载（第 {adapter.gen} 批）")
                self._cond.notify_all()
                # 加载由独立 loader 线程执行。必须在持有同一把锁的区域内
                # 进入 wait：先解锁再 wait 会与 loader 的 notify_all 竞争，
                # 一旦通知先到就永远丢失，请求会空等到超时。
                self._cond.wait(timeout=30)
                continue

        # ---- 权重已就绪：先解析基座乘法（命中则复用，不占线程等待）----
        try:
            if not self._resolve_base(record, adapter):
                return
        except Exception as exc:  # pragma: no cover - 防御性分支
            with self._lock:
                self._release_claim(adapter, record)
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                self._mark_terminal(record, FAILED, f"基座复用异常: {exc!r}")
            return

        # ---- 再锁外计算本适配器增量（模拟慢推理）----
        t0 = time.time()
        try:
            self._simulate_work(
                self.infer_delay, reason="推理", adapter_id=adapter.id
            )
            base = record.base_value
            if base is None:  # pragma: no cover - 防御性分支
                raise RuntimeError("基座结果缺失")
            delta = self.model.adapter_delta(
                record.x, weights, adapter.rank, adapter.alpha
            )
            y = base + delta
            infer_ms = round((time.time() - t0) * 1000, 1)
            result = [round(float(v), 6) for v in y[:8]]
            norm = float(np.linalg.norm(y))
            with self._lock:
                self._release_claim(adapter, record)
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                adapter.last_used_at = time.time()
                record.status = SUCCESS
                record.detail = "推理成功"
                record.result = result
                record.result_norm = norm
                record.infer_ms = infer_ms
                record.finished_at = time.time()
                record.wait_ms = round((record.started_at - record.created_at) * 1000, 1)
                record.add_event(
                    f"推理完成，用时 {infer_ms} ms；权重保留在内存"
                )
                # 释放认领后适配器可能变闲置，唤醒排队者以触发 LRU 腾挪
                self._cond.notify_all()
        except Exception as exc:  # pragma: no cover - 防御性分支
            with self._lock:
                self._release_claim(adapter, record)
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                adapter.last_used_at = time.time()
                self._mark_terminal(record, FAILED, f"推理异常: {exc!r}")
                self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # 基座乘法复用
    # ------------------------------------------------------------------ #
    def _resolve_base(self, record: RequestRecord, adapter: Adapter) -> bool:
        """取得 ``W0·x``（复用或计算）。

        返回 True 表示 ``record.base_value`` 已就绪、请求线程可以继续；
        返回 False 表示请求已结束（失败），或已登记为等待者并交还线程。
        """
        key = record.base_key
        if key is None:  # pragma: no cover - 路由层总会补齐
            key = np.ascontiguousarray(record.x, dtype=np.float64).tobytes()
            record.base_key = key

        with self._lock:
            entry = self.base_cache.get(key)
            if entry is not None and entry.state == COMPUTING:
                # 命中正在计算的项：共用这次计算，等待时不占推理线程
                self.base_cache.add_waiter(entry, record)
                record.status = BASE_WAIT
                record.base_source = BASE_JOINED
                record.detail = (
                    f"输入字节一致，共享进行中的基座乘法"
                    f"（{entry.key_label}，等 {len(entry.waiters)} 位）"
                )
                record.add_event(
                    "命中计算中的基座乘法，交还推理线程等待结果"
                )
                if record.started_at is None:
                    record.started_at = time.time()
                return False

            if entry is not None and entry.state != COMPUTING:
                # 命中已算完的项：直接复用
                self.base_cache.add_user(entry)
                record.base_value = entry.value
                record.base_source = BASE_CACHE
                record.base_cached = True
                self.base_cache.hits += 1
                record.detail = "复用缓存中的基座乘法结果"
                record.add_event(
                    f"命中基座缓存 {entry.key_label}，只计算适配器增量"
                )
                if record.started_at is None:
                    record.started_at = time.time()
                return True

            # 未命中：尝试占用一个缓存名额
            if self.base_cache.make_room():
                entry = self.base_cache.insert_computing(key, record.id)
                record.base_source = BASE_COMPUTED
                record.base_cached = True
                record.add_event("未命中基座缓存，由本请求计算 W0·x")
            else:
                # 缓存满且全部在计算/使用中：绕过缓存，照常计算
                self.base_cache.bypasses += 1
                record.base_source = BASE_BYPASS
                record.add_event("基座缓存已满且无可淘汰项，绕过缓存计算")
            if record.started_at is None:
                record.started_at = time.time()
            record.status = RUNNING

        base = self._compute_base(record, adapter)

        with self._lock:
            if not record.base_cached:
                # 绕过缓存：结果只管本请求，不写缓存
                record.base_value = base
                if base is None:
                    self._release_claim(adapter, record)
                    self._mark_terminal(
                        record, FAILED,
                        "基座乘法计算失败；本次结果不缓存，同输入可重试",
                    )
                    return False
                return True

            entry = self.base_cache.get(key)
            if base is None:
                waiters = self.base_cache.fail(entry) if entry is not None else []
                record.base_cached = False  # 条目已被移除，不再持有使用权
                self._release_claim(adapter, record)
                self._mark_terminal(
                    record, FAILED,
                    "基座乘法计算失败；等它的请求一并报错，同输入可重算",
                )
                for waiter in waiters:
                    waiter_adapter = self.adapters[waiter.adapter_id]
                    self._release_claim(waiter_adapter, waiter)
                    self._mark_terminal(
                        waiter, FAILED,
                        "所共享的基座乘法失败；同输入可重算",
                    )
                return False

            waiters = self.base_cache.finish(entry, base) if entry is not None else []
            record.base_value = base
            for waiter in waiters:
                waiter_adapter = self.adapters[waiter.adapter_id]
                waiter.base_value = base
                waiter.base_cached = True
                waiter.detail = "已取得共享的基座乘法结果，继续计算适配器增量"
                waiter.add_event(waiter.detail)
                # 重新占用一个推理线程，只算自己的适配器增量
                try:
                    self._pool.submit(self._resume_after_base, waiter, waiter_adapter)
                except RuntimeError:  # 线程池已关闭（进程退出中）
                    self._release_claim(waiter_adapter, waiter)
                    # finish() 已为等待者记了 1 个使用者，这里要退回
                    self.base_cache.release_user(waiter.base_key)
                    waiter.base_cached = False
                    self._mark_terminal(waiter, FAILED, "推理线程池已关闭")
            return True

    def set_fail_next_base(self, x, value: bool = True) -> bool:
        """模拟：让这个输入的下一次基座乘法失败一次。返回当前标记状态。"""
        canonical = np.zeros(self.model.dim_in, dtype=np.float64)
        given = np.asarray(x, dtype=np.float64).ravel()[: self.model.dim_in]
        canonical[: given.size] = given
        key = canonical.tobytes()
        with self._lock:
            if value:
                self.fail_next_base.add(key)
            else:
                self.fail_next_base.discard(key)
            return key in self.fail_next_base

    def _compute_base(self, record: RequestRecord, adapter: Adapter) -> np.ndarray | None:
        """锁外实际执行一次 ``W0 @ x``；失败返回 None（不缓存失败）。"""
        try:
            with self._lock:
                fail = record.base_key in self.fail_next_base
                if fail:
                    self.fail_next_base.discard(record.base_key)
            self._simulate_work(
                self.base_delay, reason="基座", adapter_id=adapter.id
            )
            if fail:
                raise RuntimeError("模拟基座乘法失败")
            with self._lock:
                self.base_multiplies += 1  # 真正执行 W0·x 的次数
            return self.model.base_forward(record.x)
        except Exception as exc:  # pragma: no cover - 防御性分支
            record.add_event(f"基座乘法异常: {exc!r}")
            return None

    def _resume_after_base(self, record: RequestRecord, adapter: Adapter) -> None:
        """共享基座结果的请求：线程池重新调度，只算计适配器增量。"""
        with self._lock:
            if record.status in TERMINAL_STATES:
                # 已被 shutdown 等路径提前终结：退回缓存使用权即可
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                return
            record.status = RUNNING
            record.detail = "基座结果已就绪，计算适配器增量"
        t0 = time.time()
        try:
            self._simulate_work(
                self.infer_delay, reason="推理", adapter_id=adapter.id
            )
            weights = adapter.weights
            if weights is None:  # pragma: no cover - 防御性分支
                raise RuntimeError("适配器权重已不可用")
            delta = self.model.adapter_delta(
                record.x, weights, adapter.rank, adapter.alpha
            )
            y = record.base_value + delta
            infer_ms = round((time.time() - t0) * 1000, 1)
            result = [round(float(v), 6) for v in y[:8]]
            norm = float(np.linalg.norm(y))
            with self._lock:
                self._release_claim(adapter, record)
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                adapter.last_used_at = time.time()
                record.status = SUCCESS
                record.detail = "推理成功（共用基座结果）"
                record.result = result
                record.result_norm = norm
                record.infer_ms = infer_ms
                record.finished_at = time.time()
                record.wait_ms = round(
                    (record.started_at - record.created_at) * 1000, 1
                )
                record.add_event(
                    f"适配器增量完成，用时 {infer_ms} ms；权重保留在内存"
                )
                self._cond.notify_all()
        except Exception as exc:  # pragma: no cover - 防御性分支
            with self._lock:
                self._release_claim(adapter, record)
                if record.base_cached:
                    self.base_cache.release_user(record.base_key)
                    record.base_cached = False
                adapter.last_used_at = time.time()
                self._mark_terminal(record, FAILED, f"推理异常: {exc!r}")
                self._cond.notify_all()

    def _claim(self, adapter: Adapter, record: RequestRecord) -> None:
        """调用时持锁。每个请求对每个适配器只认领一次。"""
        if not record.claimed:
            adapter.in_use += 1
            record.claimed = True

    def _release_claim(self, adapter: Adapter, record: RequestRecord) -> None:
        """调用时持锁。与 _claim 配对，保证计数平衡。"""
        if record.claimed:
            adapter.in_use -= 1
            record.claimed = False

    # ------------------------------------------------------------------ #
    # 加载 / 卸载
    # ------------------------------------------------------------------ #
    def _try_start_load_locked(self, adapter: Adapter, record: RequestRecord) -> bool:
        """调用时持锁。尝试为冷适配器腾空间并开始加载。"""
        need = adapter.nbytes

        # 空间不足：按 LRU 卸载闲置的已驻留适配器（加载中的不能动）
        if self.committed + need > self.memory_limit:
            idle = [
                a for a in self.adapters.values()
                if a.state == A_RESIDENT and a.in_use == 0
            ]
            idle.sort(key=lambda a: a.last_used_at)
            for victim in idle:
                if self.committed + need <= self.memory_limit:
                    break
                self.committed -= victim.nbytes
                victim.state = A_UNLOADED
                victim.weights = None
                self.evictions += 1
                record.add_event(
                    f"空间不足，按 LRU 卸载闲置适配器 {victim.id}，"
                    f"释放 {self._fmt(victim.nbytes)}"
                )

        if self.committed + need > self.memory_limit:
            return False  # 腾不出：调用方挂起等待

        # 加载中的权重按完整大小立即计入额度
        self.committed += need
        adapter.gen += 1
        adapter.state = A_LOADING
        adapter.weights = None
        adapter.load_attempts += 1
        adapter.last_used_at = time.time()
        threading.Thread(
            target=self._loader, args=(adapter.id, adapter.gen),
            name=f"loader-{adapter.id}-{adapter.gen}", daemon=True,
        ).start()
        return True

    def _loader(self, adapter_id: str, gen: int) -> None:
        """模拟磁盘读取 / 构造权重的慢加载过程。"""
        with self._lock:
            adapter = self.adapters[adapter_id]
            delay = self.load_delay
            will_fail = adapter_id in self.fail_next
            rank = adapter.rank
            nbytes = adapter.nbytes
            self.current_loading.add(adapter_id)
        if will_fail:
            # 一次性故障标记：只影响这一次加载
            with self._lock:
                self.fail_next.discard(adapter_id)

        try:
            self._simulate_work(
                delay, reason="加载", adapter_id=adapter_id
            )
            if will_fail:
                raise RuntimeError(f"模拟加载失败：{adapter_id} 权重文件损坏")
            weights = self.model.build_adapter(adapter_id, rank)
            actual = int(weights["A"].nbytes + weights["B"].nbytes)
            if actual != nbytes:  # pragma: no cover - 防御性校验
                raise RuntimeError(f"权重大小不一致: {actual} != {nbytes}")
        except Exception:  # noqa: BLE001 - 任何加载异常都按该批失败处理
            with self._lock:
                self.current_loading.discard(adapter_id)
                if adapter.state == A_LOADING and adapter.gen == gen:
                    adapter.state = A_FAILED
                    adapter.weights = None
                    # 注意：gen 保持单调递增、不复用，避免与 failed_gens 冲突
                    self.committed -= nbytes  # 失败立即释放额度
                    adapter.failed_gens.add(gen)
                    # 只保留最近若干批次记录，避免无界增长
                    adapter.failed_gens = set(sorted(adapter.failed_gens)[-16:])
                    self._cond.notify_all()  # 唤醒本次加载的全部等待者
            return

        with self._lock:
            if adapter.state == A_LOADING and adapter.gen == gen:
                adapter.state = A_RESIDENT
                adapter.weights = weights
                adapter.load_successes += 1
                adapter.last_used_at = time.time()
                self.current_loading.discard(adapter_id)
                self._cond.notify_all()  # 唤醒等待同一批加载的请求

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #
    def _simulate_work(self, seconds: float, *, reason: str,
                       adapter_id: str | None = None) -> None:
        """可被测试 monkeypatch 的睡眠点。"""
        if seconds > 0:
            time.sleep(seconds)

    def _mark_terminal(self, record: RequestRecord, status: str, message: str) -> None:
        """调用时持锁。"""
        record.status = status
        record.detail = message
        record.error = message if status != SUCCESS else None
        record.finished_at = time.time()
        if record.started_at is not None:
            record.wait_ms = round((record.started_at - record.created_at) * 1000, 1)
        record.add_event(message)
        self._cond.notify_all()

    @staticmethod
    def _fmt(num_bytes: int) -> str:
        value = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if value < 1024 or unit == "GiB":
                return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
            value /= 1024
        return f"{value:.1f}GiB"
