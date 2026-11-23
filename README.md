# LoRA 推理调度台（Flask + 纯 NumPy）

一个用 Flask 实现的 LoRA 推理网页：底层是**同一个 NumPy 线性模型**，挂三种大小不同的
LoRA 适配器。重点不在模型本身，而在一套**带内存额度核算的并发适配器加载调度器**，
以及**按输入向量字节复用基座乘法**的缓存。

## 功能

- 网页选择适配器、提交输入向量，查看每笔请求的状态、结果与完整事件时间线。
- 三种适配器：`lora-small`(rank 8) / `lora-medium`(rank 64) / `lora-large`(rank 256)，
  权重大小依次增大。
- 启动时设定**适配器权重内存上限**（`LORA_MEMORY_LIMIT`，基座模型常驻、不计入）。
- 可模拟**慢加载、慢推理、慢基座乘法、加载失败、基座乘法失败**（延迟可在网页运行时调整，
  失败均为一次性开关）。
- 网页实时显示当前内存占用（含加载中）、各适配器状态与**实际加载次数**、
  **实际基座乘法次数**与基座缓存明细。

## 基座乘法复用（本次新增）

模型输出是 `y = W0·x + (α/r)·B·(A·x)`。评测时同一个向量经常分别跑几款适配器，
因此把与适配器无关的 `W0·x` 按**补零后的 float64 向量原始字节**缓存：

| 规则 | 实现 |
| --- | --- |
| 命中条件 | `x.tobytes()` 完全一致（路由层已把输入补零到 `dim_in` 的 float64 向量） |
| 缓存容量 | 固定 **8 项**，`LORA_BASE_CACHE_CAPACITY` 只能调小，不能突破硬上限 |
| 计算中也占名额 | 条目在 `state=computing` 时就插入缓存并计入 8 项 |
| 命中已算完的项 | 直接复用，`base_multiplies` 不增加 |
| 命中计算中的项 | 登记为等待者并**交还推理线程**；基座算完后重新排到线程池，只算适配器增量 |
| 可淘汰条件 | 仅 `state=ready`、`waiters == 0`、`users == 0`，按最近使用时间 LRU |
| 缓存满且无法淘汰 | 新向量**绕过缓存**，在本请求线程内照常计算，不写缓存 |
| 基座计算失败 | 删除条目，发起者与全部等待者都报错；失败不缓存，下一笔同输入会重算 |
| 适配器加载失败 | 只影响该批等待者；共用同一基座结果的其他适配器请求照常推理 |
| 内存/线程上限 | 每笔请求仍按原适配器内存额度调度，仍只使用同一个请求线程池 |

等待共享基座结果**不占推理线程**，但请求对适配器权重的认领（`in_use`）在等待期间保持，
因此适配器不会被误卸载，额度核算与原来一致。

> 实测：三个适配器并发跑同一向量，`base_multiplies` 为 1（发起者 computed、后来者
> joined/cache），三条输出因适配器增量不同而各不相同，前 8 维与完整 L2 范数均正确。

## 调度规则（适配器内存）

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

加载发起者在持有调度器锁的同一区间内 `wait`，避免 loader 的 `notify_all` 先于
进入等待而丢失通知（基线版本此处存在空等 30 秒的竞态，本次已修复）。

## 目录结构

```
app/
  __init__.py     应用工厂：读环境变量、构建模型与全局唯一调度器
  config.py       大小/数值/CSV 配置解析
  model.py        NumPy 线性模型 y = W0·x + (α/r)·B·(A·x)，确定性生成三种适配器权重
  base_cache.py   基座乘法结果缓存：键=向量字节、8 项上限、等待者/使用者计数、LRU
  scheduler.py    核心：内存额度、并发加载、LRU 卸载、失败传播、基座复用（线程 + 条件变量）
  routes.py       HTTP 路由与 JSON API
  templates/index.html  单页控制台（原生 JS，1s 轮询）
wsgi.py           gunicorn / flask 入口
tests/            pytest：调度器规则 10 例 + 基座复用 12 例 + HTTP 端到端 12 例
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
| `LORA_BASE_DELAY` | `0.0` | 模拟基座乘法耗时（秒），可网页运行时改 |
| `LORA_BASE_CACHE_CAPACITY` | `8` | 基座缓存项数，只能调小，硬上限 8 |
| `LORA_FAIL_NEXT` | 空 | 逗号分隔，启动即让这些适配器的下一次加载失败 |
| `LORA_MAX_WORKERS` | `64` | 请求工作线程数（共享基座结果的等待者不占线程） |

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 控制台页面 |
| GET | `/api/state` | 内存、适配器、基座缓存、全部请求的快照 |
| GET | `/api/base` | 基座乘法次数与缓存明细 |
| POST | `/api/requests` | 提交推理 `{adapter_id, vector}`，返回 202 |
| POST | `/api/simulate/fail-next` | 设置/取消下一次加载失败 `{adapter_id, value}` |
| POST | `/api/simulate/fail-next-base` | 设置/取消某输入的下一次基座乘法失败 `{vector, value}` |
| POST | `/api/config` | 运行时调整 `{load_delay, infer_delay, base_delay}` |
| DELETE | `/api/requests` | 清理已结束的历史记录 |

`/api/state` 中的 `base` 字段包含：`multiplies`（实际 `W0·x` 次数）、`capacity`、
`size`、`computing`、`ready`、`waiting`、`in_use`、`hits`、`joins`、`misses`、
`bypasses`、`evictions`、`failures`，以及每项的 `entries`（键短哈希、状态、等待/使用数）。

## 测试

```bash
pip install pytest
PYTHONPATH=. pytest tests -q
```

覆盖：

- 适配器调度：并发只加载一次、LRU 卸载、排队等待、加载失败传播与额度释放、
  单适配器超限拒绝、慢加载期间驻留服务、冷适配器并行加载、在途加载按全额计额。
- 基座复用：跨适配器复用、纯缓存命中、补零字节一致、不同字节不共享、
  共用计算中并交还线程、8 项含计算中、满额绕过、空闲项替换、使用中不可淘汰、
  基座失败传播与重试、加载失败隔离、内存/线程上限不变。
- HTTP：提交与查询、补零命中、基座失败重试、延迟配置、错误输入等。
