"""调度器核心规则测试。"""
import threading

from conftest import make_scheduler, x_vec, wait_done, wait_state
from app.scheduler import (
    SUCCESS, FAILED, REJECTED, LOADING, PENDING, A_RESIDENT,
)


def install_gates(sched, load_blocked=(), infer_blocked=()):
    """按适配器挂门闩：加载门闩 / 推理门闩；未挂门闩的操作直接放行。

    门闩 Event 属于当前调度器实例；上一个测试泄漏的线程闭包引用的是旧
    Event，不会触碰新门闩，因此按 adapter_id 匹配也是安全隔离的。
    """
    load_gates = {aid: threading.Event() for aid in load_blocked}
    infer_gates = {aid: threading.Event() for aid in infer_blocked}
    original = sched._simulate_work

    def controlled(seconds, *, reason, adapter_id=None):
        table = load_gates if reason == "加载" else infer_gates
        gate = table.get(adapter_id)
        if gate is not None:
            gate.wait(timeout=10)
        original(seconds, reason=reason, adapter_id=adapter_id)

    sched._simulate_work = controlled
    sched.test_gates.extend(load_gates.values())
    sched.test_gates.extend(infer_gates.values())
    return load_gates, infer_gates


def wait_until(predicate, timeout=10.0, interval=0.01):
    """轮询直到谓词成立。"""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("谓词在超时前未成立")


def test_basic_inference_and_determinism(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    r1 = wait_done(sched, sched.submit("lora-small", x_vec()).id)
    r2 = wait_done(sched, sched.submit("lora-small", x_vec()).id)

    assert r1.status == SUCCESS and r2.status == SUCCESS
    assert r1.result == r2.result                       # 权重确定性
    assert sched.adapters["lora-small"].load_attempts == 1
    assert sched.adapters["lora-small"].load_successes == 1
    assert sched.committed == sizes["lora-small"]      # 用完保留在内存


def test_concurrent_requests_load_only_once(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()), load_delay=0.05)
    barrier = threading.Barrier(5)

    def worker(out):
        barrier.wait()
        out.append(wait_done(sched, sched.submit("lora-medium", x_vec()).id))

    out = []
    threads = [threading.Thread(target=worker, args=(out,)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert len(out) == 5 and all(r.status == SUCCESS for r in out)
    adapter = sched.adapters["lora-medium"]
    assert adapter.load_attempts == 1                   # 并发只加载一次
    assert sched.committed == sizes["lora-medium"]


def test_oversized_adapter_rejected(sizes):
    sched = make_scheduler(memory_limit=sizes["lora-large"] - 1)
    record = wait_done(sched, sched.submit("lora-large", x_vec()).id)

    assert record.status == REJECTED
    assert "超过内存上限" in record.error
    assert sched.committed == 0                         # 拒绝不占额度


def test_lru_eviction_when_space_insufficient(sizes):
    # 额度：大 + 小 + 8；大进来时需按 LRU 卸载闲置适配器腾挪
    limit = sizes["lora-large"] + sizes["lora-small"] + 8
    sched = make_scheduler(memory_limit=limit)

    wait_done(sched, sched.submit("lora-small", x_vec()).id)
    wait_done(sched, sched.submit("lora-medium", x_vec()).id)
    assert sched.committed == sizes["lora-small"] + sizes["lora-medium"]

    # 再用一次 small，让它成为最近使用；medium 应成为 LRU 牺牲品
    wait_done(sched, sched.submit("lora-small", x_vec()).id)
    r3 = wait_done(sched, sched.submit("lora-large", x_vec()).id)

    assert r3.status == SUCCESS
    assert sched.evictions == 1
    assert sched.adapters["lora-medium"].state != A_RESIDENT
    assert sched.adapters["lora-small"].state == A_RESIDENT
    assert sched.adapters["lora-large"].state == A_RESIDENT
    assert sched.committed == sizes["lora-small"] + sizes["lora-large"]


def test_request_waits_until_space_freed(sizes):
    # 额度只够一个适配器。medium 正在推理（in_use>0，不可卸载）时，
    # small 因装不下而排队；medium 推理结束变闲置后，LRU 卸载它，small 继续。
    sched = make_scheduler(memory_limit=sizes["lora-medium"], no_sleep=False)
    _, infer_gates = install_gates(sched, infer_blocked=["lora-medium"])

    r1_id = sched.submit("lora-medium", x_vec()).id
    wait_state(sched, r1_id, {"running"})
    assert sched.adapters["lora-medium"].in_use == 1

    r2_id = sched.submit("lora-small", x_vec()).id
    pending = wait_state(sched, r2_id, {PENDING})
    assert "等待内存腾出" in pending.detail
    # 推理中的适配器不能被卸载，额度仍被 medium 全额占用
    assert sched.committed == sizes["lora-medium"]

    infer_gates["lora-medium"].set()                      # medium 推理结束、变闲置
    r2 = wait_done(sched, r2_id)
    assert r2.status == SUCCESS
    assert sched.adapters["lora-medium"].state != A_RESIDENT
    assert sched.adapters["lora-small"].state == A_RESIDENT
    assert sched.committed == sizes["lora-small"]


def test_loading_failure_propagates_and_frees_quota(sizes):
    sched = make_scheduler(
        memory_limit=sum(sizes.values()), fail_next=["lora-small"], no_sleep=False
    )
    load_gates, _ = install_gates(sched, load_blocked=["lora-small"])

    # 门闩把加载挂住，确保 4 个并发请求都挤到同一批加载上
    ids = [sched.submit("lora-small", x_vec()).id for _ in range(4)]
    wait_until(
        lambda: sched.adapters["lora-small"].load_attempts == 1
        and all(
            next(r for r in sched._requests if r.id == rid).status == LOADING
            for rid in ids
        )
    )
    assert sched.committed == sizes["lora-small"]        # 加载中按全额计

    load_gates["lora-small"].set()                        # 放行 -> 加载失败
    records = [wait_done(sched, rid) for rid in ids]
    assert all(r.status == FAILED and "加载失败" in r.error for r in records)
    adapter = sched.adapters["lora-small"]
    assert adapter.load_attempts == 1                     # 同批并发只加载一次
    assert sched.committed == 0                           # 失败释放全部额度

    # 一次性故障：后续请求重新加载并成功
    ok = wait_done(sched, sched.submit("lora-small", x_vec()).id)
    assert ok.status == SUCCESS
    assert adapter.load_attempts == 2 and adapter.load_successes == 1
    assert sched.committed == sizes["lora-small"]


def test_slow_load_keeps_resident_adapter_serving(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()), no_sleep=False)
    load_gates, _ = install_gates(sched, load_blocked=["lora-medium"])

    # small 先驻留（不在门闩名单，加载直接放行）
    first = wait_done(sched, sched.submit("lora-small", x_vec()).id)
    assert first.status == SUCCESS

    # medium 进入慢加载，门闩关闭
    loading_id = sched.submit("lora-medium", x_vec()).id
    wait_state(sched, loading_id, {LOADING})
    assert sched.committed == sizes["lora-small"] + sizes["lora-medium"]

    # 慢加载期间，已驻留的 small 照常接单且立即成功
    resident = wait_done(sched, sched.submit("lora-small", x_vec()).id)
    assert resident.status == SUCCESS
    assert sched.adapters["lora-small"].in_use == 0

    load_gates["lora-medium"].set()                       # 放行慢加载
    loaded = wait_done(sched, loading_id)
    assert loaded.status == SUCCESS


def test_parallel_cold_loads_when_both_fit(sizes):
    sched = make_scheduler(
        memory_limit=sizes["lora-small"] + sizes["lora-medium"], no_sleep=False
    )
    load_gates, _ = install_gates(
        sched, load_blocked=["lora-small", "lora-medium"]
    )

    id_a = sched.submit("lora-small", x_vec()).id
    id_b = sched.submit("lora-medium", x_vec()).id

    # 等待两个冷适配器都进入加载态（放得下就并行加载）
    wait_state(sched, id_a, {LOADING})
    wait_state(sched, id_b, {LOADING})
    with sched._lock:
        assert sched.adapters["lora-small"].state == "loading"
        assert sched.adapters["lora-medium"].state == "loading"
        assert sched.committed == sizes["lora-small"] + sizes["lora-medium"]

    load_gates["lora-small"].set()
    load_gates["lora-medium"].set()
    assert wait_done(sched, id_a).status == SUCCESS
    assert wait_done(sched, id_b).status == SUCCESS


def test_inflight_load_counts_full_size_against_limit(sizes):
    # 额度 = 大适配器大小；小在途加载（按全额计）时，大适配器放不下必须排队
    sched = make_scheduler(
        memory_limit=sizes["lora-large"], no_sleep=False
    )
    load_gates, infer_gates = install_gates(
        sched, load_blocked=["lora-small"], infer_blocked=["lora-small"]
    )

    id_small = sched.submit("lora-small", x_vec()).id
    wait_state(sched, id_small, {LOADING})

    id_large = sched.submit("lora-large", x_vec()).id
    wait_state(sched, id_large, {PENDING})
    with sched._lock:
        assert sched.committed == sizes["lora-small"]   # 加载中按完整大小计入
        assert sched.adapters["lora-large"].state == "unloaded"

    load_gates["lora-small"].set()                       # 放行 small 加载
    # small 进入推理并被推理门闩钉住：仍占额度且不可卸载，large 继续排队
    wait_state(sched, id_small, {"running"})
    rec_large = next(r for r in sched._requests if r.id == id_large)
    assert rec_large.status == PENDING
    with sched._lock:
        assert sched.committed == sizes["lora-small"]

    infer_gates["lora-small"].set()                      # small 推理结束、变闲置
    # large 更大：small 闲置后被 LRU 卸载，large 才能加载
    large = wait_done(sched, id_large)
    assert large.status == SUCCESS
    assert sched.committed == sizes["lora-large"]
    assert sched.adapters["lora-small"].state != A_RESIDENT


def test_repeated_failures_each_notify_waiters(sizes):
    sched = make_scheduler(memory_limit=sum(sizes.values()))
    for _ in range(3):
        sched.set_fail_next("lora-medium", True)
        record = wait_done(sched, sched.submit("lora-medium", x_vec()).id)
        assert record.status == FAILED
        assert sched.committed == 0
    assert sched.adapters["lora-medium"].load_attempts == 3
