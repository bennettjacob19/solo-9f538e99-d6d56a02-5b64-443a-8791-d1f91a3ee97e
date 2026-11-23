"""基座乘法复用（按补零后 float64 向量字节缓存）的调度器测试。"""
import threading
import time

import numpy as np
import pytest

from conftest import make_scheduler, wait_done, DIM_IN, RANKS
from app.base_cache import MAX_CAPACITY
from app.scheduler import (
    SUCCESS, FAILED, BASE_WAIT, RUNNING,
    BASE_COMPUTED, BASE_CACHE, BASE_JOINED, BASE_BYPASS,
)


def gate_base(sched, hold: threading.Event):
    """把基座乘法挂住，放行由测试控制；FIFO 模式便于观察共享等待。"""
    original = sched._simulate_work

    def controlled(seconds, *, reason, adapter_id=None):
        if reason == "基座":
            hold.wait(timeout=10)
        original(seconds, reason=reason, adapter_id=adapter_id)

    sched._simulate_work = controlled
    sched.test_gates.append(hold)


def submit_many(sched, adapter_ids, x):
    return [sched.submit(a, x.copy()).id for a in adapter_ids]


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("谓词超时未成立")


def test_same_vector_across_adapters_reuses_base(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0

    ids = submit_many(sched, ["lora-small", "lora-medium", "lora-large"], x)
    records = [wait_done(sched, rid) for rid in ids]

    assert all(r.status == SUCCESS for r in records)
    assert sched.base_multiplies == 1
    # 三条结果与逐个独立计算完全一致（浮点相等）
    model = sched.model
    for rec, aid in zip(records, ["lora-small", "lora-medium", "lora-large"]):
        adapter = model.build_adapter(aid, RANKS[aid])
        expect = model.forward(x, adapter, RANKS[aid], 16.0)
        assert rec.result_norm == pytest.approx(float(np.linalg.norm(expect)), rel=1e-12)
        assert rec.result == [round(float(v), 6) for v in expect[:8]]
    assert len({tuple(r.result) for r in records}) == 3  # 适配器增量确实不同


def test_cache_hit_does_not_recompute_base(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0
    wait_done(sched, sched.submit("lora-small", x).id)
    assert sched.base_multiplies == 1

    # 适配器已驻留，第二笔应当纯命中缓存
    second = wait_done(sched, sched.submit("lora-medium", x).id)
    assert second.base_source == BASE_CACHE
    assert sched.base_multiplies == 1
    assert sched.base_cache.hits == 1

    # 不同向量必须重新计算
    other = x.copy()
    other[0] += 1.0
    wait_done(sched, sched.submit("lora-small", other).id)
    assert sched.base_multiplies == 2


def test_padded_vectors_with_same_bytes_share_base(sizes):
    """短向量补零后字节一致，应视为同一个缓存项。"""
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    short = np.array([1.0, 2.0], dtype=np.float64)
    padded = np.zeros(DIM_IN, dtype=np.float64)
    padded[:2] = [1.0, 2.0]

    wait_done(sched, sched.submit("lora-small", short).id)
    hit = wait_done(sched, sched.submit("lora-medium", padded).id)

    assert hit.base_source == BASE_CACHE
    assert sched.base_multiplies == 1


def test_different_bytes_do_not_share(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    a = np.arange(DIM_IN, dtype=np.float64)
    b = a.copy()
    b[-1] += 1e-12  # 字节不同
    wait_done(sched, sched.submit("lora-small", a).id)
    wait_done(sched, sched.submit("lora-small", b).id)
    assert sched.base_multiplies == 2
    assert sched.base_cache.size == 2


def test_joining_inflight_base_releases_infer_thread(sizes):
    """命中计算中的项：等结果时交还线程，且只算一次基座。"""
    sched = make_scheduler(memory_limit=sum(sizes.values()), no_sleep=False)
    hold = threading.Event()
    gate_base(sched, hold)
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0

    ids = submit_many(sched, ["lora-small", "lora-medium", "lora-large"], x)
    # 发起者可能是任意一笔请求：等齐 2 个等待者后，再按 base_source 定位
    wait_until(lambda: sched.base_cache.stats()["waiting"] == 2)
    records_by_id = {r.id: r for r in sched._requests}
    owner = next(r for r in records_by_id.values() if r.id in ids
                 and r.base_source == BASE_COMPUTED)
    joiner = next(r for r in records_by_id.values() if r.id in ids
                  and r.base_source == BASE_JOINED)
    assert owner.status == RUNNING
    assert joiner.status == BASE_WAIT
    assert joiner.detail.startswith("输入字节一致")
    assert sched.base_multiplies == 0  # 还在基座延迟里，尚未算完

    hold.set()
    records = [wait_done(sched, rid) for rid in ids]
    assert all(r.status == SUCCESS for r in records)
    assert sched.base_multiplies == 1
    # 等待共享结果的请求也拿到了正确的独立适配器输出
    assert len({tuple(r.result) for r in records}) == 3


def test_joiners_do_not_consume_extra_workers(sizes):
    """等待共享基座结果的请求交还线程，只留一个线程在算基座。"""
    sched = make_scheduler(
        memory_limit=sum(sizes.values()), max_workers=2, no_sleep=False
    )
    hold = threading.Event()
    gate_base(sched, hold)
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0

    ids = submit_many(sched, ["lora-small", "lora-medium", "lora-large"], x)
    # 2 个 worker：1 个算基座；其余请求加入等待后交还线程，最终都成为等待者
    wait_until(lambda: sched.base_cache.stats()["waiting"] == 2)

    hold.set()
    records = [wait_done(sched, rid) for rid in ids]
    assert all(r.status == SUCCESS for r in records)
    assert sched.base_multiplies == 1


def test_cache_capacity_is_eight_including_computing(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()), no_sleep=False)
    assert sched.base_cache.capacity == 8
    hold = threading.Event()
    gate_base(sched, hold)

    ids = []
    for i in range(MAX_CAPACITY):
        x = np.zeros(DIM_IN, dtype=np.float64)
        x[0] = i + 1
        ids.append(sched.submit("lora-small", x).id)

    wait_until(lambda: sched.base_cache.size == MAX_CAPACITY)
    with sched._lock:
        assert sched.base_cache.stats()["computing"] == MAX_CAPACITY
        # 8 个都在计算，都不可淘汰
        assert all(not e.evictable for e in sched.base_cache.entries.values())

    # 第 9 个不同向量：缓存满且无法淘汰 -> 绕过缓存
    x9 = np.zeros(DIM_IN, dtype=np.float64)
    x9[0] = 99.0
    r9_id = sched.submit("lora-small", x9).id
    wait_until(lambda: next(
        r for r in sched._requests if r.id == r9_id
    ).base_source == BASE_BYPASS)
    assert sched.base_cache.bypasses == 1
    assert sched.base_cache.size == MAX_CAPACITY

    hold.set()
    records = [wait_done(sched, rid) for rid in ids]
    assert all(r.status == SUCCESS for r in records)
    assert wait_done(sched, r9_id).status == SUCCESS
    # 8 个缓存计算 + 1 个绕过 = 9 次真实基座乘法
    assert sched.base_multiplies == MAX_CAPACITY + 1


def test_ready_idle_entry_is_evicted_for_new_vector(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))

    ids = []
    for i in range(MAX_CAPACITY):
        x = np.zeros(DIM_IN, dtype=np.float64)
        x[0] = i + 1
        ids.append(sched.submit("lora-small", x).id)
    for rid in ids:
        wait_done(sched, rid)
    assert sched.base_cache.size == MAX_CAPACITY

    x9 = np.zeros(DIM_IN, dtype=np.float64)
    x9[0] = 99.0
    rec = wait_done(sched, sched.submit("lora-small", x9).id)
    assert rec.status == SUCCESS
    assert sched.base_cache.evictions == 1
    assert sched.base_cache.bypasses == 0
    assert sched.base_cache.size == MAX_CAPACITY
    assert sched.base_multiplies == MAX_CAPACITY + 1


def test_in_use_entry_not_evicted(sizes):
    """已算完但仍有请求在用的条目不能被淘汰。"""
    sched = make_scheduler(memory_limit=sum(sizes.values()), no_sleep=False)
    infer_hold = threading.Event()
    blocked_once = {"done": False}
    original = sched._simulate_work

    def controlled(seconds, *, reason, adapter_id=None):
        # 只把第一笔推理挂在门闩上，其余请求正常结束
        if reason == "推理" and not blocked_once["done"]:
            blocked_once["done"] = True
            infer_hold.wait(timeout=10)
        original(seconds, reason=reason, adapter_id=adapter_id)

    sched._simulate_work = controlled
    sched.test_gates.append(infer_hold)

    x = np.zeros(DIM_IN, dtype=np.float64)
    x[0] = 1.0
    held_id = sched.submit("lora-small", x).id
    wait_until(lambda: next(
        r for r in sched._requests if r.id == held_id
    ).status == RUNNING)
    with sched._lock:
        entry = sched.base_cache.get(
            next(r for r in sched._requests if r.id == held_id).base_key
        )
        assert entry is not None and entry.state == "ready"
        assert entry.users == 1 and not entry.evictable

    # 再填 7 个不同向量，全部算完后变成可淘汰的空闲条目
    for i in range(MAX_CAPACITY - 1):
        y = np.zeros(DIM_IN, dtype=np.float64)
        y[0] = 10.0 + i
        sched.submit("lora-small", y)
    wait_until(lambda: sched.base_cache.size == MAX_CAPACITY
               and sched.base_cache.stats()["computing"] == 0)

    # 新向量：唯一仍在使用的第 1 项不可淘汰，替换其余 7 个空闲项之一
    z = np.zeros(DIM_IN, dtype=np.float64)
    z[0] = 777.0
    z_id = sched.submit("lora-small", z).id
    wait_until(lambda: next(
        r for r in sched._requests if r.id == z_id
    ).base_source in {BASE_COMPUTED, BASE_BYPASS})
    assert sched.base_cache.bypasses == 0
    assert sched.base_cache.evictions == 1
    with sched._lock:
        held_entry = sched.base_cache.get(
            next(r for r in sched._requests if r.id == held_id).base_key
        )
        assert held_entry is not None  # 被使用的条目仍在缓存中

    infer_hold.set()
    assert wait_done(sched, held_id).status == SUCCESS
    assert wait_done(sched, z_id).status == SUCCESS


def test_base_failure_fails_all_waiters_and_allows_retry(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()), no_sleep=False)
    hold = threading.Event()
    gate_base(sched, hold)
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0
    sched.set_fail_next_base(x, True)

    ids = submit_many(sched, ["lora-small", "lora-medium", "lora-large"], x)
    wait_until(lambda: sched.base_cache.stats()["waiting"] == 2)

    hold.set()
    records = [wait_done(sched, rid) for rid in ids]
    assert all(r.status == FAILED for r in records)
    assert "基座乘法" in records[0].error
    assert all("基座乘法" in r.error for r in records)
    assert sched.base_cache.size == 0          # 失败条目已移除
    assert sched.base_cache.failures == 1
    assert sched.base_multiplies == 0          # 失败不算一次有效乘法

    # 下一笔同输入可重算并成功
    ok = wait_done(sched, sched.submit("lora-small", x).id)
    assert ok.status == SUCCESS
    assert sched.base_multiplies == 1
    assert sched.base_cache.size == 1


def test_base_failure_does_not_affect_other_adapters_batches(sizes):
    """一组适配器加载失败不应影响共用基座结果的其他适配器。"""
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    x = np.arange(DIM_IN, dtype=np.float64) + 1.0

    # 先让 medium 的加载失败，同时 small 用同一向量正常推理
    sched.set_fail_next("lora-medium", True)
    medium_id = sched.submit("lora-medium", x).id
    small_id = sched.submit("lora-small", x).id

    medium = wait_done(sched, medium_id)
    small = wait_done(sched, small_id)
    assert medium.status == FAILED and "加载失败" in medium.error
    assert small.status == SUCCESS
    assert sched.base_multiplies == 1

    # small 已缓存基座；重试 medium 加载成功后直接复用基座
    medium_retry = wait_done(sched, sched.submit("lora-medium", x).id)
    assert medium_retry.status == SUCCESS
    assert medium_retry.base_source == BASE_CACHE
    assert sched.base_multiplies == 1


def test_memory_and_thread_limits_still_enforced(sizes):
    """缓存复用不改变适配器内存额度与线程池上限的语义。"""
    limit = sizes["lora-medium"]   # medium 放得下，small 放不下时按 LRU 淘汰
    sched = make_scheduler(memory_limit=limit, max_workers=4)
    assert sched.memory_limit == limit
    assert sched._pool._max_workers == 4

    x = np.arange(DIM_IN, dtype=np.float64) + 1.0
    wait_done(sched, sched.submit("lora-small", x).id)
    # 同一输入换 medium：适配器装不下，必须按原规则 LRU 腾挪
    medium_id = sched.submit("lora-medium", x).id
    rec = wait_done(sched, medium_id, timeout=10)
    assert rec.status == SUCCESS
    assert sched.adapters["lora-small"].state != "resident"
    assert sched.committed == sizes["lora-medium"]
    assert sched.base_multiplies == 1  # 缓存命中，没有因换适配器重算基座
