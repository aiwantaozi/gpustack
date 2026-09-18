# GPUStack PD 分离 —— 对标业界六家的差距验证报告

| 项 | 值 |
|---|---|
| 方法 | 以 [orchestrator-comparison.md](.cache/plan/pd-disaggregation/orchestrator-comparison.md) 与 [orchestrator-problems.md](.cache/plan/pd-disaggregation/orchestrator-problems.md) 记录的**六家已知缺陷**为反向测试清单，逐条在 GPUStack 上验证同类问题是否存在 |
| 对标对象 | Dynamo · llm-d · AIBrix · KServe · RBG · Kthena |
| 目标环境 | http://192.168.50.15:9000（前端 build `6bcaa3c`），cluster 2 `x1`，1 worker × 2×RTX 4090 |
| 测试路径 | API + 源码交叉核对（代码结论一律配黑盒实证） |
| 执行时间 | 2026-09-01 |
| 结论 | **3 条确凿缺陷 · 3 条文档漂移 · 3 项设计观察 · 4 项优于业界 · 5 项能力缺口** |

> 本轮不重复上一份 [E2E 报告](gpu-e2e-pd-disaggregation-20260831-192811.md) 已覆盖的功能验证，只做"与业界对位"的发散验证。

---

## 一、测试用例是怎么设计出来的

对比文档给出了一个可复用的判据（§三 3.1）：

> 🔑 PD 的抽象放在哪一层不重要，**重要的是「配对关系」有没有一个地方能被校验**。

以及一条贯穿六家的模式（§七 风险矩阵结尾）：

> 🔴 级别的风险里，「静默」出现在 llm-d 三条、AIBrix 两条、KServe 一条、Kthena 四条、RBG 两条 —— 形态高度一致：**一个能力在某个条件下不生效，系统继续正常服务，没有任何信号**。

据此把六家的缺陷折成 5 类可执行的探针，每类都在 GPUStack 上找对应位置：

| 探针 | 业界原型 | 在 GPUStack 上问什么 |
|---|---|---|
| **P1 枚举即能力** | Kthena `lmcache`→HTTPConnector、`mooncake`→NIXL 别名；AIBrix `mooncake` 两方法 `return nil` | `DisaggregationSpec` 的每个字段，设成非默认值后**真的改变行为吗** |
| **P2 配对键可校验性** | Kthena `GroupKey` 自由字符串、无校验、跨组同值则所有 P/D 落一个桶 | 两个组同时跑，router 的成员表会不会串 |
| **P3 静默失效可见性** | Kthena `kv_transfer_params` 缺失只 warning → 200 正常返回的退化 | 有判定机制吗？**阈值抓得住部分退化吗** |
| **P4 资源与端口治理** | RBG 端口分配不回收（KEP 自承）；gang 与 dependencies 静默死锁 | 端口删除后会回收吗；成组准入算得准吗 |
| **P5 失败处理粒度** | Kthena 换对重试（六家唯一）vs llm-d 重试同一 host vs Dynamo 直接失败 | 成员死了，请求还能不能成 |

---

## 二、确凿缺陷

### 🔴 缺陷 1：`disaggregation.readiness` 的 `"all"` 是不生效的枚举值

**这正是对比文档「三条不该学」的第三条**：*半成品不该出现在用户可选的枚举里 —— 要么不暴露，要么选中时明确报「未实现」*。

| | |
|---|---|
| 声明 | [models.py:649](gpustack/schemas/models.py#L649) `readiness: Literal["any_per_role", "all"] = "any_per_role"` |
| 消费点 | **零**。全仓库 grep 不到任何读取 `disaggregation.readiness` 的代码 |
| 实际判定 | [controllers.py:2152-2156](gpustack/server/controllers.py#L2152) 硬编码 `status.ready == 0`，即"每角色至少一个 ready"（= `any_per_role` 语义），与字段取值无关 |

**黑盒实证**：建组时设 `readiness: "all"`，扩容 prefill 到 4（资源只够 3 个）：

```
disagg 回读: {"mode":"vllm-nixl","readiness":"all","kv_load_failure_policy":"recompute","router_kind":"not-a-real-router-kind"}
state=running  role_status={"prefill":{"desired":4,"ready":3},"decode":{"desired":1,"ready":1},"router":{"desired":1,"ready":1}}
degradations=['ratio_unmet']  ready_targets=1     ← 路由目标 ACTIVE，组被判为可服务
```

`"all"` 的语义要求"全部成员 ready 才算组就绪"，此刻 prefill 3/4 应当**不可服务**；实际 `state=running` 且 `ready_targets=1`，与 `any_per_role` 逐字节一致。

**业界对位**：Kthena 的 `MinRoleReplicas` 是六家唯一把"组就绪"做成 per-role 可配的（steal list #23），且对比文档专门提醒抄它时"能力探测失败必须响亮"。我们这个字段像是同一个意图的入口，但停在了 schema 层。

**建议**：二选一 —— 实现它（`readiness=="all"` 时把判定改成 `status.ready < status.desired` 即不可服务），或从 `Literal` 里摘掉 `"all"` 只留单值。当前状态是最坏的一种：用户能设、能存、能读回，唯独不起作用。

---

### 🔴 缺陷 2：`disaggregation.router_kind` 字段零消费点

| | |
|---|---|
| 声明 | [models.py:651-652](gpustack/schemas/models.py#L651) `router_kind: Optional[str] = None`，注释 `"""None derives it from mode."""` |
| 消费点 | **零**。全仓库无任何读取 |
| 暴露面 | 出现在每个 PD 模型的 API 响应里（`"router_kind": null`） |

**黑盒实证**：设 `router_kind: "not-a-real-router-kind"` → HTTP 200 接受、持久化、原样回读，组照常起来、router 照常运行、无任何校验错误或告警。

一个**任意字符串都被接受且完全无效**的字段，比不存在更糟：它让读 API 的人以为可以在这里换 router 实现。

---

### 🟠 缺陷 3：`kv_load_failure_policy` 只在 4 个内置 mode 中的 1 个生效

比前两条更隐蔽：它**不是不生效，是部分生效**。

| mode | 是否引用 `{{kv_load_failure_policy}}` |
|---|---|
| `vllm-nixl` | ✅ 两处（[pd-modes.yaml:201, 222](gpustack/assets/pd-modes.yaml#L201)，prefill 与 decode 各一） |
| `sglang-mooncake` | ❌ |
| `sglang-nixl` | ❌ |
| `vllm-ascend-mooncake` | ❌ |

**黑盒实证**（vllm-nixl，证明它在这个 mode 上确实有效）：

```
decode 实例 injected_backend_parameters:
  --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_consumer",
                        "kv_load_failure_policy":"recompute"}    ← 设的 recompute 进去了
```

在另外三个 mode 上设置同一字段，不会进入任何引擎参数，且**没有任何提示**。

**为什么这条重要**：对比文档 §九「上线前行动清单」第 5 条专门讲这个字段的选型 ——

> `kv_load_failure_policy` 选型：`fail`（默认，推荐）会 500 但不拖累 decode；`recompute` 会静默劣化。选 `recompute` 就必须加「D 侧 prefill 重算次数」的指标，否则问题不可见

也就是说这是一个**需要用户认真权衡**的字段。在 SGLang 两条 mode 上，用户做完权衡、设了值、以为拿到了 recompute 兜底，实际什么都没发生。

---

## 三、文档与实现漂移（对标「文档骗人」专栏）

对比文档 §七把这类单列为"本轮最有价值的新发现类别 —— 不是能力缺失，而是文档与实现不一致，会直接导致误判"，并给了方法论：**判断一个能力有没有，应该查 API 定义 + 消费点 + sample，而不是读设计文档**。用这条方法查我们自己：

### 🔴 漂移 1：断路器阈值，文档说 2，实际 4 个 mode 全是 10

[docs/user-guide/pd-disaggregation.md:225-226](docs/user-guide/pd-disaggregation.md#L225)：

> The built-in modes **narrow the upstream router's circuit breaker so a dead role member is taken out of rotation after two failed requests rather than ten**; a router you supply keeps whatever defaults it ships with.

实际 [pd-modes.yaml](gpustack/assets/pd-modes.yaml) 四个内置 mode 全部是 `--cb-failure-threshold "10"`（行 364 / 542 / 680 / 851）—— 即**上游默认值，没有任何 narrow**。

而且 pd-modes.yaml 自己的注释把回退原因记得很清楚（行 343-362）：

> 🔴 Was "2", raised to upstream's own default after a live failure (2026-08-28, SGLang 1P1D). Under GPU contention a handful of requests timed out; at threshold 2 that opened the prefill circuit, **and it never closed again** — every later request was fast-failed in ~0.17s with `No available prefill workers`… **The group was down and nothing else in the product said so**: all three instances RUNNING, the model `state: running`.

⇒ 实现是对的（回退有充分实证），**是用户文档没跟着改**。这句话现在承诺了一个不存在的行为，而且方向恰好相反 —— 用户会以为故障成员 2 次失败就被摘除。

### 🟠 漂移 2：前缀路由，文档说"不可用/round-robin"，prefill 侧实际是 cache_aware

[docs/user-guide/pd-disaggregation.md:236](docs/user-guide/pd-disaggregation.md#L236)：

> Prefix-aware routing is not available; the router balances round-robin.

实际三个 mode 的 router 命令是：

```
--prefill-policy cache_aware      ← 前缀感知（approximate radix tree over request text）
--decode-policy  round_robin
```

pd-modes.yaml 的注释还专门论证了为什么 decode 用 round_robin 而 prefill 用 cache_aware（行 265-274）。⇒ 准确说法应是"**decode 侧**没有前缀感知路由"，而不是整体不可用。

**这一条与业界的对位很有意思**：对比文档 §4.1 记录了"decode 侧要不要前缀亲和"是六家 2:2 的真实分歧，而我们的实现（prefill 前缀感知 + decode 负载均衡）恰好站在 Dynamo / llm-d well-lit-path 那一边，还写了理由。**这是个可以拿出来讲的设计决策，却被文档写成了"没有这个能力"。**

### 🟢 漂移 3（反方向）：成员失效检测，实现优于文档

[docs/user-guide/pd-disaggregation.md:228-231](docs/user-guide/pd-disaggregation.md#L228)：

> A member that dies while the group is idle is **noticed on the next request, not before it**… The first requests after an idle period are **retried and then routed around** the dead member.

**实测**（3P1D 组，删掉一个 running prefill 后立刻连发 8 次）：

```
after-kill-0  HTTP 200  0.31s  ...prefill_addr_192.168.50.15:40045...
after-kill-1  HTTP 200  0.09s  ...prefill_addr_192.168.50.15:40045...
after-kill-2  HTTP 200  0.08s  ...prefill_addr_192.168.50.15:40028...
...
结果码: [200,200,200,200,200,200,200,200]     被杀的 40017 一次都没被选中
```

零失败、零重试代价（首个请求 0.31s，其余 <0.1s）。说明是**控制面主动摘除 peer**（`pd_membership.py` 的 reconcile），而不是数据面靠断路器发现。

⚠️ **但这不能证明文档错了** —— 我测的是"实例被删除"（控制面知情），文档描述的是"成员进程崩溃"（控制面不知情），两条路径不同。文档描述的场景需要在 worker 主机上 kill 进程才能复现，本轮未做。

---

## 四、设计观察（不是 bug，但值得讨论）

### 观察 1：`AGGREGATED_RATIO = 0.01` 让有效性判定只能抓"全停"，抓不住"部分退化"

[pd_metrics.py:44](gpustack/server/pd_metrics.py#L44) `AGGREGATED_RATIO = 0.01`，判定逻辑（[:191-206](gpustack/server/pd_metrics.py#L191)）：

```python
if requests <= 0:                          return "idle"
if transfers / requests < AGGREGATED_RATIO: return "aggregated"
return "effective"
```

即 **100 个请求里只要有 1 个发生了 KV 传输，就判 `effective`**。

这个阈值抓得住"PD 完全没生效"，抓不住业界反复出现的那一类：**3P1D 里一个 prefill 的 NIXL agent 坏了** —— 此时 ratio ≈ 0.67，判定仍是 `effective`，而三分之一的请求已经在退化。

**业界对位**：这套判定本身是我们相对六家的差异化（Kthena 在"PD 到底生效没有"上是零，llm-d 要用户自己监控指标比值）。但对比文档 §六 #9 的纪律是"**fail-open 也必须可见**"，同理，一个判定如果只在 99% 失效时才报警，中间地带就是不可见的。

**建议**：把单一阈值改成分档（例如 `<0.01` = aggregated、`0.01~0.8` = degraded、`>0.8` = effective），或按 per-role 拆开算——现有 `roles` 字段已经带了 per-role 数据，缺的只是把比值也做成 per-role。

### 观察 2：显存记账低估，PD 组会放大它

实测三组数据：

```
GPUStack 记账:  alloc = 30.9 GB / 卡
实际 GPU 占用:  used  = 42.8 GB / 卡        ← 差 12 GB
按记账剩余 20.6 GB 部署 3 个 7.7 GB 成员 → decode 实例 CUDA OOM 退出
  日志: torch.OutOfMemoryError: GPU 0 has a total capacity of 47.37 GiB
        of which 147.50 MiB is free
```

上一轮也遇到过同类问题的另一种形态（两个角色都取默认 `--gpu-memory-utilization=0.9`，先起的 decode 吃掉整卡，prefill 报 `No suitable workers`）。

**对位设计目标**：[short-term-design.md F5](.cache/plan/pd-disaggregation/short-term-design.md) §3.7 的目标是"成组放置：一次解出全组、**装得下就一定放得下**"。两轮实测都没达到 —— 一次是排不上，一次是排上了但引擎 OOM。

⚠️ **公平地说**：这两次都有环境竞争（用户的模型同时在跑），且显存记账是 GPUStack 的通用问题、不是 PD 引入的。但 **PD 把它放大了**：一个 1P1D 组是 3 个成员、3P1D 是 5 个，同一批卡上的并发放置数是非 PD 部署的 3-5 倍，估算误差被同比放大。

### 观察 3：启动失败的可操作性不均

同一个组里两种失败，信息量差距很大：

| 失败 | `state_message` |
|---|---|
| router `exit 127` | 一整段可操作解释：*"The image does not contain the command this member was started with… A managed router is the usual case: the mode's recipe launches a router binary that the engine's runner image does not ship, and the fix is to give the router role an image of its own that carries it."* |
| decode `exit 1`（CUDA OOM） | **`Error (exit code 1)`** —— 原因只在实例日志里 |

**对位**：[X2 §3.3](.cache/plan/pd-disaggregation/short-term-design.md) 的目标是把 `IndexError` / 永远 `starting` 这类翻译成可操作提示。OOM 是最常见的部署失败，反而没有被翻译。日志里的原文（`torch.OutOfMemoryError … 147.50 MiB is free`）本身已经足够清楚，缺的只是把它提到 `state_message`。

---

## 五、优于业界的点（本轮验证通过）

### ✅ 1. 配对域由系统身份划分，不是用户填的自由字符串 —— 结构上不存在 Kthena 的头号风险

Kthena 的 `GroupKey` 是用户填的 label key，对比文档列为其头号风险之一：

> 🟠 **`GroupKey` 无任何校验** —— 填一个跨组同值的 label 会让所有 P/D 落进一个桶、**配对保证形同虚设，而且完全静默**

**实测**（两个 PD 组同时运行，直接查各自 router 的成员注册表）：

```
router :40016 (模型 57 t1-qwen3-0.6b, 2P2D)
  workers 全部 model_id="t1-qwen3-0.6b"  (40059 prefill / 40029 decode / 40013 decode / …)

router :40041 (模型 60 e2etest-pd-ports, 1P1D)
  workers: [{url:40022, model_id:"e2etest-pd-ports", worker_type:"decode"},
            {url:40034, model_id:"e2etest-pd-ports", worker_type:"prefill"}]
  stats: {"total_workers":2, "healthy_workers":2, "total_models":1}
```

两组零串扰，每条注册都带系统生成的 `model_id`。而且 `total_models: 1` 正好满足对比文档对这类机制的要求 ——「必须在校验层证明这个键真的划分了集合，并**暴露「当前有几个配对域」**」。

### ✅ 2. 端口回收正常 —— 不是 RBG 那种只增不回收

RBG 的已知缺陷：*端口分配器随机分配 + **不回收**，KEP 自己承认跨 controller 不保证唯一；长期运行的集群会漏*。

**实测**（删组后立即重建）：

```
第 1 代占用: [40000, 40007, 40008, 40009, 40010, 40012, 40028, 40040, 40042, 40045, 40046, 40058]
删除 → 重建
第 2 代占用: [40000, 40006, 40022, 40034]        ← 复用了 40000，新端口落在低位而非从 40059 递增
```

端口回到池中可再分配，无单调递增泄漏迹象。

### ✅ 3. 两跳协议参数由系统注入，用户没机会配错

对标 Kthena #28（对比文档说这条"把『忘了配 → 静默退化成聚合式』这一半成因**从架构上消灭**"）。我们同样由系统合成 `--kv-transfer-config`（实测 `injected_backend_parameters` 可见），且比 Kthena 多一层：**用户手写冲突参数会在准入期被拒**（上一轮 B-19/B-20 已验证）。

### ✅ 4. PD 有效性判定存在且端到端可用

六家里 Kthena 明确是"零"（*连 `kv_transfer_params` 缺失都只打一条 warning，30 个指标里没有一个能发现它*），llm-d 要用户自己监控 `llm_d_epp_disagg_decision_total` 比值。我们有 `status` + `kv_transfers_per_request` + per-role 指标 + UI 展示，上一轮实测拿到 `status=effective / ratio=1.0 / decode 侧 cached_prompt_tokens 100%`。

⚠️ 但见【观察 1】：阈值让它只能抓极端情况。

---

## 六、能力缺口（六家有、我们没有，不构成缺陷）

| # | 能力 | 谁有 | 我们的状态 |
|---|---|---|---|
| 1 | **组就绪 per-role 数量可配**（`MinRoleReplicas`） | Kthena（六家唯一） | `readiness` 字段像是这个意图的入口，但未实现（见缺陷 1） |
| 2 | **跨角色比值护栏**（`RatioConstraint`，先抬后降） | Kthena（六家唯一） | 无。设计标为二期（F5） |
| 3 | **Desired / Final 双副本数分别可见** | Kthena（六家唯一） | `role_status` 是 `desired/ready`（期望/就绪），不是"指标算出的/护栏后的"。二期做扩缩时才需要 |
| 4 | **失败重试粒度 = 一对 P/D** | Kthena（六家唯一） | router 是 `--retry-max-retries 3`，未验证重试时是否换对。P 池部分不健康时，换对是唯一能自愈的粒度 |
| 5 | **路由层拓扑就近** | Dynamo / llm-d / AIBrix | 我们和 Kthena / RBG 一样只在**放置层**（`gather`），路由层不知道拓扑。对比文档 §3.3 指出这两层"必须都做，只做前者路由器照样跨域配对" |

---

## 七、本轮测试用例清单

已执行 15 条，未执行 4 条（列出原因，便于后续补齐）。

| 用例 | 探针 | 结果 |
|---|---|---|
| G-01 `readiness="all"` 被接受并持久化 | P1 | ✅ 接受（缺陷 1 前半） |
| G-02 `readiness="all"` 改变组就绪判定 | P1 | ❌ **未生效**（缺陷 1） |
| G-03 `router_kind` 任意值被接受 | P1 | ✅ 接受、零效果（缺陷 2） |
| G-04 `kv_load_failure_policy=recompute` 注入 vllm-nixl | P1 | ✅ 生效 |
| G-05 同字段在 SGLang mode 的引用 | P1 | ❌ **不引用**（缺陷 3，源码判定） |
| G-06 断路器阈值 vs 用户文档 | — | ❌ **漂移**（文档 2 / 实际 10 × 4 mode） |
| G-07 前缀路由能力 vs 用户文档 | — | ❌ **漂移**（文档"不可用" / 实际 prefill cache_aware） |
| G-08 两组并存时 router 成员表隔离 | P2 | ✅ 零串扰，带 `model_id`，`total_models=1` |
| G-09 配对域数量可观测 | P2 | ✅ router `/workers` 的 `stats.total_models` |
| G-10 有效性判定阈值对部分退化的灵敏度 | P3 | ⚠️ **0.01 过宽**（观察 1，源码判定） |
| G-11 端口删除后回收复用 | P4 | ✅ 复用 40000，无单调递增 |
| G-12 成组准入的显存估算准确性 | P4 | ⚠️ 记账 30.9GB vs 实际 42.8GB → decode OOM（观察 2） |
| G-13 成员被删后请求可用性 | P5 | ✅ 8/8 200，无一落到死成员 |
| G-14 router 在多 prefill 间轮转 | P5 | ✅ 3 次基线请求命中 3 个不同 prefill |
| G-15 降级下继续服务 | P5 | ✅ prefill 3/4 时 `running` + `ratio_unmet` + `ready_targets=1` |
| — | P5 | ⛔ **进程崩溃**（非删除）后的断路器路径：需在 worker 主机 kill 进程，本轮未做 |
| — | P3 | ⛔ **人为制造部分退化**（让一个 prefill 的 connector 失效）验证判定灵敏度：无干净的注入手段 |
| — | P4 | ⛔ **gang 与 dependencies 死锁**（RBG R1 同形问题）：需 k8s 上资源不足 + 原子准入场景 |
| — | P1 | ⛔ **SGLang mode 下 `recompute` 的黑盒确认**：需拉 SGLang 镜像，本轮以源码引用判定 |

---

## 八、建议的处理顺序

| 优先级 | 动作 | 理由 |
|---|---|---|
| **1** | 修用户文档两处漂移（断路器阈值、前缀路由） | 零成本，且当前文案承诺了不存在的行为。前缀路由那条还把一个**有理有据的设计决策**写成了"没有这个能力" |
| **2** | `readiness` 与 `router_kind` 二选一处理：实现或摘掉 | 对比文档明写这类是"三条不该学"之一；`readiness` 若实现，正好补上六家里只有 Kthena 有的能力 |
| **3** | `kv_load_failure_policy` 在不支持的 mode 上准入期拒绝或提示 | 比前两条更隐蔽：它在一个 mode 上有效，用户没有理由怀疑另外三个无效 |
| **4** | 有效性判定阈值分档 / per-role 化 | 这是我们相对六家的差异化能力，但当前只能抓极端情况。per-role 数据已经有了 |
| **5** | 把 OOM 一类的失败原因提到 `state_message` | router exit 127 的文案已经证明这条路走得通 |
| 6 | 成组准入的显存估算（跨轮两次实测都没兜住） | 工作量大，且是 GPUStack 通用问题，PD 只是放大器 |

---

## 九、清理情况

| 资源 | 处理 |
|---|---|
| `e2etest-pd-decl`（id 58，OOM 失败的 2P1D） | ✅ 已删 |
| `e2etest-pd-decl`（id 59，readiness 实验组，扩到 4P1D） | ✅ 已删 |
| `e2etest-pd-ports`（id 60，端口回收实验组） | ✅ 已删 |
| 自动创建的同名 route | ✅ 随模型级联删除 |
| **用户资源** | **未改动**：`t1-qwen3-0.6b`(57) 全程保持 running，其余 6 个模型保持其 pending 状态 |

复核：`e2etest` 前缀的模型 / 路由 / 实例均为 0。

---

*报告生成时间：2026-09-01 11:37*
