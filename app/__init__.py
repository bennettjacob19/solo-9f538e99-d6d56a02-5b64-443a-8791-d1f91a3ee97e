"""Flask 应用工厂。"""
from __future__ import annotations

import os

from flask import Flask

from .model import ADAPTER_SPECS, LinearLoRAModel
from .scheduler import LoRAScheduler
from .routes import register_routes
from .config import parse_size, parse_float, parse_csv, parse_int


def create_app(config: dict | None = None) -> Flask:
    """创建 Flask 应用并构建全局唯一的调度器（单进程多线程模型）。"""
    config = config or {}

    dim_in = parse_int(config.get("LORA_DIM_IN", os.environ.get("LORA_DIM_IN", "1024")))
    dim_out = parse_int(config.get("LORA_DIM_OUT", os.environ.get("LORA_DIM_OUT", "1024")))
    memory_limit = parse_size(
        config.get("LORA_MEMORY_LIMIT", os.environ.get("LORA_MEMORY_LIMIT", "4.5MiB"))
    )
    load_delay = parse_float(
        config.get("LORA_LOAD_DELAY", os.environ.get("LORA_LOAD_DELAY", "2.0"))
    )
    infer_delay = parse_float(
        config.get("LORA_INFER_DELAY", os.environ.get("LORA_INFER_DELAY", "1.0"))
    )
    fail_next = parse_csv(
        config.get("LORA_FAIL_NEXT", os.environ.get("LORA_FAIL_NEXT", ""))
    )
    max_workers = parse_int(
        config.get("LORA_MAX_WORKERS", os.environ.get("LORA_MAX_WORKERS", "64"))
    )
    base_cache_capacity = parse_int(
        config.get("LORA_BASE_CACHE_CAPACITY",
                   os.environ.get("LORA_BASE_CACHE_CAPACITY", "8"))
    )
    base_delay = parse_float(
        config.get("LORA_BASE_DELAY", os.environ.get("LORA_BASE_DELAY", "0"))
    )

    model = LinearLoRAModel(dim_in=dim_in, dim_out=dim_out)
    adapter_metas = []
    for spec in ADAPTER_SPECS:
        meta = dict(spec)
        meta["nbytes"] = model.adapter_nbytes(spec["rank"])
        adapter_metas.append(meta)

    scheduler = LoRAScheduler(
        model=model,
        adapter_metas=adapter_metas,
        memory_limit=memory_limit,
        load_delay=load_delay,
        infer_delay=infer_delay,
        fail_next=fail_next,
        max_workers=max_workers,
        base_cache_capacity=base_cache_capacity,
        base_delay=base_delay,
    )

    app = Flask(__name__)
    app.scheduler = scheduler  # type: ignore[attr-defined]
    register_routes(app)
    return app
