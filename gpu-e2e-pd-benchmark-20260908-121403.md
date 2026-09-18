# GPUStack E2E 测试报告 —— PD 分离 §6.3 矩阵 + benchmark 改动验证

目标：http://localhost:9000（本地 UI → 远程 server `172.22.32.14`，版本 `v0.0.0 / git_commit 6bc0d48`）
认证：password  ·  测试路径：both（API + UI）  ·  深度：quick（未启动任何部署）
结果：**7 通过 / 0 失败 / 11 无法执行（缺前置） / 3 观察**

## 环境实况（发现阶段）

| 集群 | worker | 卡 | 状态 |
|---|---|---|---|
| ascend | asc-w28 `192.168.13.2` · asc-w22 `192.168.13.3` | 8 × 昇腾 each | ready |
| nvidia | gpuhost409048g1 `192.168.50.15` | 2 × NVIDIA | ready |
| Default Cluster | VM-32-14-ubuntu `172.22.32.14` | 0 | ready |

🔴 **所有模型当前 `replicas=0`（全停机）**：`b1-pd`(ascend, vllm-ascend-mooncake, prefill/decode/router) · `qwen3.5-4b`(nvidia, vllm-nixl, 三角色) · `qwen3.5-4b-agg`(nvidia, 聚合基线) · `b1-pd-copy`。
⇒ 需要真实流量的用例本轮**没有执行**，改为核对历史 run 留下的事实。

## 已验证（基于真实历史 run + UI 实操）

| # | 验的是什么 | 路径 | 结果 | 证据 |
|---|---|---|---|---|
| V1 | benchmark 目标解析到 **router**（PD 组不能指名成员） | API | ✅ | run 9–13 的 `model_instance_name` 全是 `b1-pd-router-971sn` / `qwen3.5-4b-router-k4l7v` |
| V2 | 快照收**全组**且带 role | API | ✅ | run 13 快照三成员：`prefill@asc-w28` `decode@asc-w28` `router@asc-w22` |
| V3 | 组的卡数不再是 router 的 0 | API | ✅ | run 13 `gpus=8`、`gpu_summary=910B2x8`、`workers=[asc-w28, asc-w22]` |
| V4 | `spec_digest` 记录代次 | API | ✅ | run 13 `sha1:16f89fdf…` |
| V5 | `target_mode` 往返 + route 模式记路由名 | API | ✅ | run 16 `target_mode=route`、`snapshot.route_name=qwen3.5-4b-agg` |
| V6 | route 模式端到端能跑完 | API | ✅ | run 16 `completed`、peak_rate 21.0（run 15 是修 `/health` 之前的 error） |
| V7 | **PD 角色 GPU 选择器反显**（本轮修复） | UI | ✅ | `b1-pd` 编辑抽屉 → Roles → Prefill → 手动：显示 `asc-w28 / 910B2 +3` + `GPUs per Replica 4`（修复前为空） |
| V8 | 形态在上、目标在下，且**联动** | UI | ✅ | 顺序 `Cluster → Target type → Benchmark target`；切到 Route 后下方从 cascader 变为 `Route` 下拉 |
| V9 | route 空态文案 | UI | ✅ | 「No route fronts a servable LLM in this cluster…」（全停机下正确为空） |

## 🔴 发现：route 模式在高速率下自身会失败，会污染对比

run 16（`qwen3.5-4b-agg`，1×4090，route 模式）与 run 13（`b1-pd`，昇腾 PD，instance 模式）的**错误分布形状不同**：

| rate | route 模式 err / total | instance 模式 err / total |
|---|---|---|
| ≤8 | 0 / 460 | 0 / 400 |
| 16–18 | 0 / 480 | 1 / 540 |
| 20–24 | 3→13→22→59 | 2–4（恒定） |
| 28–32 | 142 → **220 / 960** | 3 / 900（rate 48 仍是 3） |

错误类型是 `ConnectTimeout` 与 `HTTP 500 Internal server error` —— 这是**代理侧**的失败形状（引擎过载会排队/变慢，不会拒连），与设计里写明的「route 把 server 的代理算进链路，高并发下瓶颈可能是代理」一致，但比"慢"更严重：它在丢请求。

⚠️ **对 PD vs 聚合对比的直接影响**：≥20 rps 正是对比开始有意思的区间，而 route 模式在这里自带错误底噪。
⚠️ 归因保留：两次 run 的模型与硬件不同（4090 聚合 vs 910B2 PD），不能只凭这两条断定全是代理的锅 —— 但错误**类型**指向代理。建议下一步：同一个部署分别用 instance / route 各跑一次，同机对照。

## 无法执行的用例（缺前置，未假装跑过）

| 用例 | 缺什么 |
|---|---|
| T1/T2 冒烟、T13 生命周期 | 需要启动部署（当前全停机），属高成本操作，未经确认不执行 |
| T3 xPyD 配比 | 需要 2P1D / 1P2D 部署 |
| T4/T4b/T5 带宽三档与 TCP 回落 | 需要 Qwen3-8B + 跨机 + 人为破坏 RDMA |
| T6/T7/T8/T9/T10 | 需要 thinking 模型 / MoE dp2×tp2 / GLM-4.7-Flash / 异构 TP 部署 |
| T11 缓存共存 | 需要先部署共享 KV Cache |
| T12/T16/T17/T18/T18b | 需要 Qwen1.5-MoE / gpt-oss-20b / Nemotron / Qwen3.5-9B，且部分只在 NVIDIA、部分只在昇腾 |

## 用户体验观察

1. **英文文案冲突（已修）**：形态字段原为 `Target`、目标字段为 `Benchmark target`，英文里更泛的词落在了更具体的字段上。已改为 `Target type`（commit `403f21c4`）；其余四语种本就区分。
2. **全停机时两个下拉都空**：instance 模式按 `replicas>0` 过滤、route 模式按 servable 过滤，停机集群下都为空。route 有明确空态文案，**instance 侧没有** —— 用户会看到一个没有任何选项、也不说为什么的下拉。
3. **run 16 的 `state_message` 说 `errored=459`，而行上的 `request_errored=13`**：前者是全部测点求和，后者是代表点。数字不一致但都不算错，读报告的人容易困惑。

## 清理

未创建任何资源；打开过 `b1-pd` 编辑抽屉与 benchmark 创建抽屉，**均以 Cancel 退出，无保存**。浏览器 task space 已关闭。

---

# 第二轮：实际启动部署后的测试（2026-09-08 下午）

启动了 `b1-pd`（昇腾 1P1D，prefill TP4 @cann0-3 / decode TP4 @cann4-7 / router，均在 asc-w28+asc-w22）与 `qwen3.5-4b-agg`（NVIDIA 1×4090 聚合基线）。

## T1 / T15 —— 昇腾 PD 冒烟：✅ 全绿

| 断言 | 结果 | 证据 |
|---|---|---|
| F1 角色建出来并 ready | ✅ | `state=running`，`role_status` 三个角色各 1/1；`ready_replicas=3`（RUNNING 实例纯计数，D26 口径） |
| F2 pd_mode 派生 | ✅ | `vllm-ascend-mooncake`，backend `0.20.2-ascend-router-custom` |
| F3 端口分配 | ✅ | prefill HTTP 40045 + `kv_port` 40004–40007（**4 个 = 卡数**）；decode 40029 + 40000–40003；router 40004 + 独立 `prometheus` 40000（避开硬编码 29000） |
| F4 两跳 | ✅ | 请求 id 编码双跳地址：`chatcmpl-___prefill_addr_192.168.13.2:40045___decode_addr_192.168.13.2:40029_…` —— 地址正是 API 报的那两个端口，端口分配→引擎实际使用闭环 |
| §5.9 ① P 侧每请求 +1 token | ✅ | prefill 日志 `prompt throughput 29.1 tok/s` vs `generation 1.2 tok/s` |
| §5.9 ② D 侧 KV transfer | ✅ | `mooncake_hybrid_connector.py:648 KV cache transfer … took 1.48 ms (1 groups, 1 blocks)`，TP0/TP1/TP3 各 rank 均有 |
| 答案正确 | ✅ | 12 并发请求全 200（7.5–8.7 s，thinking 模型），单请求答案推理方向正确 |

## 准入负向用例（`replicas=0` 创建，只跑校验不占卡）

| 用例 | 期望 | 实际 | 结论 |
|---|---|---|---|
| T10 反向：NIXL，P TP2 / D TP1 | 拒绝 | **400** | ✅ |
| T10 正向：NIXL，P TP1 / D TP2 | 放行 | 200 | ✅（已删除） |
| 昇腾 mooncake，P TP2 / **D TP4** | 拒绝 | **400**「gathers each decode rank's KV from prefill ranks…」 | ✅ 方向按 recipe 分流生效 |
| 昇腾 mooncake，**P TP4** / D TP2（华为参考形态） | 放行 | 200 | ✅ 旧的一刀切 NIXL 规则会误杀这个（已删除） |
| `max_model_len` 8192 vs 4096 | 拒绝 | **400** | ✅ |
| hybrid KV cache manager 分歧 | 拒绝 | **400** | ✅ |
| **T8：非 MoE + decode `--data-parallel-size=2`** | **拒绝** | **200 被接受** | 🔴 **缺口**：矩阵要求准入期拦截，实现里没有这条规则（已删除测试模型） |

## 🔴 关键发现：route 模式在本拓扑下不能用于性能对比

同一部署（`qwen3.5-4b-agg`，1×4090）、同一 profile、同一数据集，唯一差别是路径：

| rate | route TTFT / 错误 | instance TTFT / 错误 |
|---|---|---|
| 1 | 1398 ms / 0 | **96 ms** / 0 |
| 8 | 1928 ms / 0 | **113 ms** / 0 |
| 16 | 6713 ms / 0 | 5187 ms / 0 |
| 20 | 8515 ms / **3** | 10933 ms / **0** |
| 24 | 11095 ms / **59** | 16806 ms / **0** |
| 32 | 13658 ms / **220** | 28682 ms / **0** |
| **peak_rate** | **21** | **16** |

三条结论：

1. **低负载下 route 多出 ~1.3 s 纯延迟**（96→1398 ms）。不是代理 CPU，是**网络拓扑**：压测容器在 worker（私网 192.168.50.15），route 模式把每个请求发到云上 server（43.128.120.44）再由它代理回同一台 worker —— 每请求两趟公网。
2. **错误全在代理侧**：同一引擎直连时 rate 32 仍 0 错，经代理时 220 错（ConnectTimeout / HTTP 500）。
3. 🔴 **最危险的一条**：route 在高负载下 TTFT **反而更低**（13.7 s vs 28.7 s）、`peak_rate` **反而更高**（21 vs 16）—— 因为它在**丢请求**，活下来的自然等得少。不看错误列会得出「route 更快」的相反结论。

⇒ **PD vs 聚合对比必须用 instance 模式**，或者把压测客户端放到与 server 同网的位置。当前 UI 的 help 文案只说了「高并发下代理可能成为瓶颈」，**没说它会让指标看起来更好**，这一点要补。

## 环境限制：指标类功能在本拓扑下全空

`GET /v2/models/8/pd-metrics` 返回 `status=unmeasurable`、`roles={}`、`request_count_source=none`。根因不是代码：server 上的 Prometheus 只有 **2 个 target**（自己 up，`localhost:10161` down），**没有任何 worker/引擎 target** —— worker 在私网、server 在云上，Prometheus 是拉模式够不着。
🟢 值得肯定的是：接口回的是「测不到」+ 原因文案（Mooncake 无 transfer 计数器 → 走引擎 token 口径），**没有编一个看起来正常的 0**（D33 的设计意图）。

## 仍未执行

T3（需 2P1D/1P2D 改配比）· T4/T4b（需 Qwen3-8B + 跨机带宽三档）· T5（人为破坏 RDMA，**按指示不跑**）· T6（thinking + 短 ISL 反例）· T7（MoE dp2×tp2）· T9/T11/T12/T16/T17/T18/T18b（需 GLM-4.7-Flash / 共享缓存 / Qwen1.5-MoE / gpt-oss / Nemotron / Qwen3.5-9B 等未部署模型）。

## 本轮遗留

- **仍在运行**：`b1-pd`（占 asc-w28 8 卡）· `qwen3.5-4b-agg`（占 4090 1 卡）—— 未停，便于继续做对比；要停请说。
- **测试产物**：benchmark `e2etest-agg-instance-0908-1243`（id=17），是上表 instance 列的数据来源，保留作证据。
- 已删除：3 个准入测试模型（id 12/13/14，均 `replicas=0`，未占卡）。
