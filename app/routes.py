"""HTTP 路由：网页与 JSON API。"""
from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("main", __name__)


def _scheduler():
    return current_app.scheduler


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/state")
def get_state():
    return jsonify(_scheduler().snapshot())


@bp.post("/api/requests")
def submit_request():
    payload = request.get_json(silent=True) or {}
    adapter_id = payload.get("adapter_id")
    scheduler = _scheduler()

    if not adapter_id or adapter_id not in scheduler.adapters:
        return jsonify({"error": f"未知适配器: {adapter_id!r}"}), 400

    # 输入向量：支持 JSON 数组 / 逗号分隔字符串 / 单个数字（自动补零）
    raw = payload.get("vector", payload.get("x", []))
    dim = scheduler.model.dim_in
    try:
        if isinstance(raw, str):
            values = [float(p) for p in raw.split(",") if p.strip()]
        elif isinstance(raw, (int, float)):
            values = [float(raw)]
        else:
            values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return jsonify({"error": "输入向量必须是数字数组或逗号分隔的数字"}), 400

    if len(values) > dim:
        return jsonify({"error": f"输入维度 {len(values)} 超过模型维度 {dim}"}), 400
    if not values:
        values = [0.0]
    x = np.zeros(dim, dtype=np.float64)
    x[: len(values)] = np.asarray(values, dtype=np.float64)

    record = scheduler.submit(adapter_id, x)
    return jsonify({
        "id": record.id,
        "status": record.status,
        "detail": record.detail,
        "base_key": record.base_key.hex()[:24] if record.base_key else None,
    }), 202


@bp.post("/api/simulate/fail-next")
def simulate_fail_next():
    payload = request.get_json(silent=True) or {}
    adapter_id = payload.get("adapter_id")
    value = bool(payload.get("value", True))
    try:
        adapter = _scheduler().set_fail_next(adapter_id, value)
    except KeyError:
        return jsonify({"error": f"未知适配器: {adapter_id!r}"}), 400
    return jsonify({"id": adapter.id, "will_fail_next": adapter.id in _scheduler().fail_next})


@bp.post("/api/simulate/fail-next-base")
def simulate_fail_next_base():
    """模拟某个输入的下一次基座乘法失败（一次性）。"""
    payload = request.get_json(silent=True) or {}
    scheduler = _scheduler()
    raw = payload.get("vector", payload.get("x", []))
    if isinstance(raw, str):
        raw = [p for p in raw.split(",") if p.strip()]
    try:
        values = [float(v) for v in np.asarray(raw, dtype=np.float64).ravel()]
    except (TypeError, ValueError):
        return jsonify({"error": "vector 必须是数字数组"}), 400
    if len(values) > scheduler.model.dim_in:
        return jsonify({"error": "输入维度超过模型维度"}), 400
    marked = scheduler.set_fail_next_base(values, bool(payload.get("value", True)))
    return jsonify({"will_fail_next": marked})


@bp.get("/api/base")
def get_base():
    scheduler = _scheduler()
    with scheduler._lock:
        return jsonify({
            "multiplies": scheduler.base_multiplies,
            "delay": scheduler.base_delay,
            **scheduler.base_cache.stats(),
        })


@bp.post("/api/config")
def update_config():
    payload = request.get_json(silent=True) or {}
    kwargs = {}
    if "load_delay" in payload:
        try:
            kwargs["load_delay"] = float(payload["load_delay"])
        except (TypeError, ValueError):
            return jsonify({"error": "load_delay 必须是数字（秒）"}), 400
    if "infer_delay" in payload:
        try:
            kwargs["infer_delay"] = float(payload["infer_delay"])
        except (TypeError, ValueError):
            return jsonify({"error": "infer_delay 必须是数字（秒）"}), 400
    if "base_delay" in payload:
        try:
            kwargs["base_delay"] = float(payload["base_delay"])
        except (TypeError, ValueError):
            return jsonify({"error": "base_delay 必须是数字（秒）"}), 400
    return jsonify(_scheduler().update_config(**kwargs))


@bp.delete("/api/requests")
def clear_requests():
    _scheduler().clear_requests()
    return jsonify({"ok": True})


def register_routes(app):
    app.register_blueprint(bp)
