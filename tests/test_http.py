"""Flask HTTP 接口端到端测试（线程内测试客户端 + 零延迟调度器）。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.scheduler import SUCCESS, FAILED, REJECTED, TERMINAL_STATES


def _app(tmp_path):
    app = create_app({
        "LORA_DIM_IN": "16",
        "LORA_DIM_OUT": "12",
        "LORA_MEMORY_LIMIT": "1MiB",
        "LORA_LOAD_DELAY": "0",
        "LORA_INFER_DELAY": "0",
    })
    app.scheduler._simulate_work = lambda *a, **k: None
    return app


def _wait(client, rid):
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        state = client.get("/api/state").get_json()
        rec = next(r for r in state["requests"] if r["id"] == rid)
        if rec["status"] in TERMINAL_STATES:
            return rec
        time.sleep(0.01)
    raise AssertionError("请求超时未结束")


def test_index_page():
    app = _app(None)
    client = app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "LoRA" in page.get_data(as_text=True)


def test_submit_and_state_flow():
    app = _app(None)
    client = app.test_client()

    resp = client.post("/api/requests", json={
        "adapter_id": "lora-small", "vector": [1, 2, 3],
    })
    assert resp.status_code == 202
    rid = resp.get_json()["id"]

    rec = _wait(client, rid)
    assert rec["status"] == SUCCESS
    assert len(rec["result"]) == 8

    state = client.get("/api/state").get_json()
    assert state["memory"]["committed"] > 0
    adapter = next(a for a in state["adapters"] if a["id"] == "lora-small")
    assert adapter["load_attempts"] == 1 and adapter["load_successes"] == 1


def test_bad_inputs_rejected():
    app = _app(None)
    client = app.test_client()

    assert client.post("/api/requests", json={"adapter_id": "nope"}).status_code == 400
    too_long = [1.0] * 999
    resp = client.post("/api/requests", json={
        "adapter_id": "lora-small", "vector": too_long,
    })
    assert resp.status_code == 400


def test_oversized_adapter_rejected_over_http():
    app = create_app({
        "LORA_DIM_IN": "16", "LORA_DIM_OUT": "12",
        "LORA_MEMORY_LIMIT": "1",  # 1 字节，任何适配器都超限
        "LORA_LOAD_DELAY": "0", "LORA_INFER_DELAY": "0",
    })
    app.scheduler._simulate_work = lambda *a, **k: None
    client = app.test_client()
    resp = client.post("/api/requests", json={"adapter_id": "lora-large"})
    rid = resp.get_json()["id"]
    rec = _wait(client, rid)
    assert rec["status"] == REJECTED


def test_fail_next_simulation_endpoint():
    app = _app(None)
    client = app.test_client()

    resp = client.post("/api/simulate/fail-next",
                       json={"adapter_id": "lora-medium", "value": True})
    assert resp.get_json()["will_fail_next"] is True

    rid = client.post("/api/requests",
                      json={"adapter_id": "lora-medium"}).get_json()["id"]
    rec = _wait(client, rid)
    assert rec["status"] == FAILED and "加载失败" in rec["error"]

    state = client.get("/api/state").get_json()
    assert state["memory"]["committed"] == 0
    adapter = next(a for a in state["adapters"] if a["id"] == "lora-medium")
    assert adapter["will_fail_next"] is False  # 一次性标记已消费


def test_runtime_delay_config_update():
    app = _app(None)
    client = app.test_client()
    resp = client.post("/api/config", json={"load_delay": 4.5, "infer_delay": 0.25})
    cfg = resp.get_json()
    assert cfg["load_delay"] == 4.5 and cfg["infer_delay"] == 0.25

    bad = client.post("/api/config", json={"load_delay": "abc"})
    assert bad.status_code == 400


def test_clear_history():
    app = _app(None)
    client = app.test_client()
    rid = client.post("/api/requests",
                      json={"adapter_id": "lora-small"}).get_json()["id"]
    _wait(client, rid)
    client.delete("/api/requests")
    state = client.get("/api/state").get_json()
    assert state["requests"] == []


def test_same_vector_reuses_base_over_http():
    app = _app(None)
    client = app.test_client()
    vector = [1, 2, 3, 4]

    ids = []
    for adapter_id in ("lora-small", "lora-medium", "lora-large"):
        ids.append(client.post("/api/requests", json={
            "adapter_id": adapter_id, "vector": vector,
        }).get_json()["id"])
    records = [_wait(client, rid) for rid in ids]

    assert all(r["status"] == SUCCESS for r in records)
    assert all(len(r["result"]) == 8 and r["result_norm"] is not None
               for r in records)
    state = client.get("/api/state").get_json()
    assert state["base"]["multiplies"] == 1
    assert state["base"]["size"] == 1
    assert {r["base_source"] for r in records} <= {"computed", "cache", "joined"}

    base = client.get("/api/base").get_json()
    assert base["multiplies"] == 1 and base["capacity"] == 8


def test_padded_vector_cache_key_matches():
    app = _app(None)
    client = app.test_client()

    rid1 = client.post("/api/requests", json={
        "adapter_id": "lora-small", "vector": [1, 2],
    }).get_json()["id"]
    _wait(client, rid1)
    # 显式补零到模型维度，字节一致
    rid2 = client.post("/api/requests", json={
        "adapter_id": "lora-medium", "vector": [1, 2] + [0] * 14,
    }).get_json()["id"]
    rec2 = _wait(client, rid2)

    assert rec2["status"] == SUCCESS
    assert rec2["base_source"] == "cache"
    state = client.get("/api/state").get_json()
    assert state["base"]["multiplies"] == 1


def test_base_failure_endpoint_and_retry():
    app = _app(None)
    client = app.test_client()
    vector = [1, 2, 3]

    resp = client.post("/api/simulate/fail-next-base",
                       json={"vector": vector, "value": True})
    assert resp.get_json()["will_fail_next"] is True

    rid = client.post("/api/requests", json={
        "adapter_id": "lora-small", "vector": vector,
    }).get_json()["id"]
    rec = _wait(client, rid)
    assert rec["status"] == FAILED and "基座乘法" in rec["error"]

    state = client.get("/api/state").get_json()
    assert state["base"]["multiplies"] == 0
    assert state["base"]["size"] == 0
    assert state["base"]["failures"] == 1

    # 下一笔同输入可重算
    rid2 = client.post("/api/requests", json={
        "adapter_id": "lora-small", "vector": vector,
    }).get_json()["id"]
    rec2 = _wait(client, rid2)
    assert rec2["status"] == SUCCESS
    state = client.get("/api/state").get_json()
    assert state["base"]["multiplies"] == 1 and state["base"]["size"] == 1


def test_base_failure_endpoint_accepts_csv_string():
    app = _app(None)
    client = app.test_client()
    resp = client.post("/api/simulate/fail-next-base",
                       json={"vector": "1, 2, 3", "value": True})
    assert resp.status_code == 200 and resp.get_json()["will_fail_next"] is True

    bad = client.post("/api/simulate/fail-next-base",
                      json={"vector": [1.0] * 999})
    assert bad.status_code == 400


def test_base_delay_config_update():
    app = _app(None)
    client = app.test_client()
    cfg = client.post("/api/config", json={"base_delay": 0.5}).get_json()
    assert cfg["base_delay"] == 0.5

    bad = client.post("/api/config", json={"base_delay": "abc"})
    assert bad.status_code == 400
