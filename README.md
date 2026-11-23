# LoRA 推理调度台（Flask + 纯 NumPy）

一个用 Flask 实现的 LoRA 推理网页：底层是**同一个 NumPy 线性模型**，挂三种大小不同的
LoRA 适配器。重点不在模型本身，而在一套**带内存额度核算的并发适配器加载调度器**。

## 功能

- 网页选择适配器、提交输入向量，查看每笔请求的状态、结果与完整事件时间线。
- 三种适配器：`lora-small`(rank 8) / `lora-medium`(rank 64) / `lora-large`(rank 256)，
  权重大小依次增大。
- 启动时设定**适配器权重内存上限**（`LORA_MEMORY_LIMIT`，基座模型常驻、不计入）。
- 可模拟**慢加载、慢推理、加载失败**（延迟可在网页运行时调整，失败为一次性开关）。
- 网页实时显示当前内存占用（含加载中）、各适配器状态与**实际加载次数**。

## 调度规则（核心）

| 规则 | 实现方式 |
| --- | --- |
| 启动设内存上限 | `LORA_MEMORY_LIMIT`，支持 `4.5MiB` / `1024` 等写法 |
| 加载中的权重按**完整大小**计入 | `_try_start_load_locked` 一旦开始加载立即 `committed += nbytes` |
| 同一适配器并发请求只加载一次 | 后来者复用同一加载批次（`gen`），在条件变量上等待，不重复加载 |
| 用完的权重留在内存 | 推理结束不释放，仅更新 LRU 时间戳 |
| 空间不足才卸载闲置权重 | 仅从 `state=resident 且 in_use=0` 的适配器里按 LRU 卸载 |
| 暂时装不下的请求等待 | 进入 `pending`，额度腾出后被唤醒重试 |
| 慢加载期间驻留适配器仍接单 | 驻留适配器请求直接推理；放得下的冷适配器可并行开始加载 |
| 单个适配器超限直接拒绝 | 提交时判定 `nbytes > limit` → `rejected`，不占任何额度 |
| 加载失败 | 同批所有等待请求收到错误；立即释放额度；一次性故障标记供后续请求重试 |

### 一个关键的并发正确性细节

请求在**发起或加入加载的那一刻就认领（`in_use += 1`）**，而不是等权重驻留后才认领。
这样刚加载好、但请求尚未真正开始推理的适配器，绝不会因为处于 `in_use == 0` 的瞬间窗口
而被另一个排队请求错误地 LRU 卸载。认领在请求成功/失败时严格配对释放（见
`LoRAScheduler._claim / _release_claim`）。

## 目录结构

```
app/
  __init__.py     应用工厂：读环境变量、构建模型与全局唯一调度器
  config.py       大小/数值/CSV 配置解析
  model.py        NumPy 线性模型 y = W0·x + (α/r)·B·(A·x)，确定性生成三种适配器权重
  scheduler.py    核心：内存额度、并发加载、LRU 卸载、失败传播（线程 + 条件变量）
  routes.py       HTTP 路由与 JSON API
  templates/index.html  单页控制台（原生 JS，1s 轮询）
wsgi.py           gunicorn / flask 入口
tests/            pytest：调度器并发规则 10 例 + HTTP 端到端 7 例
Dockerfile
docker-compose.yml
```

> 调度器是**进程内状态**，因此生产部署必须单 worker、多线程
> （compose 中 gunicorn `--workers 1 --threads 64`）。

## 快速开始

### docker compose（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

### 本地运行

```bash
pip install -r requirements.txt
LORA_MEMORY_LIMIT=4.5MiB LORA_LOAD_DELAY=2 LORA_INFER_DELAY=1 python wsgi.py
# 或：gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 64 wsgi:app
```

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LORA_MEMORY_LIMIT` | `4.5MiB` | 适配器权重内存上限（加载中按全额计入） |
| `LORA_DIM_IN` / `LORA_DIM_OUT` | `1024` / `1024` | 线性模型维度 |
| `LORA_LOAD_DELAY` | `2.0` | 模拟加载耗时（秒），可网页运行时改 |
| `LORA_INFER_DELAY` | `1.0` | 模拟推理耗时（秒），可网页运行时改 |
| `LORA_FAIL_NEXT` | 空 | 逗号分隔，启动即让这些适配器的下一次加载失败 |
| `LORA_MAX_WORKERS` | `64` | 请求工作线程数 |

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 控制台页面 |
| GET | `/api/state` | 内存、适配器、全部请求的快照 |
| POST | `/api/requests` | 提交推理 `{adapter_id, vector}`，返回 202 |
| POST | `/api/simulate/fail-next` | 设置/取消下一次加载失败 `{adapter_id, value}` |
| POST | `/api/config` | 运行时调整 `{load_delay, infer_delay}` |
| DELETE | `/api/requests` | 清理已结束的历史记录 |

## 测试

```bash
pip install pytest
PYTHONPATH=. pytest tests -q
```

覆盖：并发只加载一次、LRU 卸载、排队等待、加载失败传播与额度释放、单适配器超限拒绝、
慢加载期间驻留服务、冷适配器并行加载、在途加载按全额计额等。
