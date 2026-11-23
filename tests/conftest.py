"""pytest 公共夹具：小维度模型 + 无延迟调度器。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from app.model import LinearLoRAModel
from app.scheduler import LoRAScheduler, TERMINAL_STATES

DIM_IN, DIM_OUT = 16, 12
RANKS = {"lora-small": 2, "lora-medium": 4, "lora-large": 8}

# 记录所有测试创建的调度器，便于统一清理门闩与线程池
_LIVE_SCHEDULERS: list = []


def make_scheduler(memory_limit, *, load_delay=0.0, infer_delay=0.0,
                   fail_next=None, no_sleep=True, max_workers=32,
                   base_cache_capacity=8, **kwargs):
    model = LinearLoRAModel(dim_in=DIM_IN, dim_out=DIM_OUT)
    metas = []
    for adapter_id, rank in RANKS.items():
        metas.append({
            "id": adapter_id, "rank": rank, "alpha": 16.0,
            "nbytes": model.adapter_nbytes(rank),
        })
    scheduler = LoRAScheduler(
        model=model, adapter_metas=metas, memory_limit=memory_limit,
        load_delay=load_delay, infer_delay=infer_delay,
        fail_next=fail_next, max_workers=max_workers,
        base_cache_capacity=base_cache_capacity, **kwargs,
    )
    if no_sleep:
        scheduler._simulate_work = lambda *a, **k: None
    _LIVE_SCHEDULERS.append(scheduler)
    return scheduler


@pytest.fixture(autouse=True)
def _cleanup_schedulers():
    yield
    while _LIVE_SCHEDULERS:
        sched = _LIVE_SCHEDULERS.pop()
        sched.shutdown()


@pytest.fixture
def sizes():
    model = LinearLoRAModel(dim_in=DIM_IN, dim_out=DIM_OUT)
    return {aid: model.adapter_nbytes(r) for aid, r in RANKS.items()}


def x_vec(n=DIM_IN):
    return np.arange(n, dtype=np.float64) + 1.0


def wait_state(scheduler, rid, states, timeout=10.0):
    """轮询等待某请求进入指定状态集合；超时则输出调度器快照便于定位。"""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        with scheduler._lock:
            record = next(r for r in scheduler._requests if r.id == rid)
            if record.status in states:
                return record
        time.sleep(0.01)
    with scheduler._lock:
        dump = {
            "requests": [
                (r.id, r.adapter_id, r.status, r.detail, r.base_source,
                 r.claimed, r.gen, list(r.events))
                for r in scheduler._requests
            ],
            "adapters": [
                (a.id, a.state, a.in_use, a.gen, a.load_attempts)
                for a in scheduler.adapters.values()
            ],
            "committed": scheduler.committed,
            "base": scheduler.base_cache.stats(),
        }
    raise AssertionError(
        f"请求 {rid} 未在 {timeout}s 内进入 {states}；快照：{dump}"
    )


def wait_done(scheduler, rid, timeout=10.0):
    return wait_state(scheduler, rid, TERMINAL_STATES, timeout)
