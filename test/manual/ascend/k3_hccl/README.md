# Kimi K3 / DSPARK HCCL 数值对照

基线分支：`a5-k3-0908`；诊断分支：`k3-hccl`。
本目录用于人工启动服务、离线取证；不需要服务器连接外网。

## 对照边界

- 问题描述是四机 TP32/DP1；提供的脚本实际是四机 TP64/DP4，
  开启 DP attention / DP LM head。启动前确认采用哪个配置，日志 manifest
  会保存实际 TP/DP、模型、量化和推测参数，不能按日志文件名推断拓扑。
- 单机裁层按 TP8/DP1。AIV 和 CCU_SCHED 两次必须使用**同一裁层产物、
  同一 draft 权重、同一 tokenizer、同一拓扑、同一请求**。
  不要直接将 TP8 和 TP32/64 张量逐元素比较。
- TP8 复现可以继续缩小问题；TP8 正常不能排除跨机算法、消息阈值、
  TP 分片或混部问题。裁层后的接受长度也不是完整模型精度指标。
- 保留 K3 原层号、位置和必要状态；不要仅修改层数后把异常接受率归因于通信。

## 启动前修改原脚本

在所有参与节点安装/同步此分支，使原脚本的 `SGLANG_PATH` 指向其 `python/`。
同时保存 `git rev-parse HEAD`、裁层配置、权重来源，以及 CANN/HCCL、驱动、
固件、torch/torch_npu 版本。对照时只改变 HCCL 模式和日志目录。

将原脚本中写死的 HCCL 模式替换为：

```bash
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
```

保留提供脚本的其他优化开关，包括 FIAS V2/NZ、MLAPO、重计算、
DeepEP 和 DSPARK block size 7。第一轮数值对照在 **两组** 启动命令中，
用以下选项替换 `--cuda-graph-bs ...`：

```bash
--cuda-graph-backend-decode disabled \
--cuda-graph-backend-prefill disabled
```

不要启用 torch.compile。本诊断若发现启用图/compile，会报错退出，
不会悄悄更改执行配置。CPU 张量快照会同步生产流、改变 overlap 时序；
此轮只用于数值定位，不能据此证明原图模式或异步路径没有问题。
DeepEP 内部传输/量化并不等于普通 HCCL collective；其调用边界单独记录。

关图检查读取执行器使用的 `get_exec().graph.cuda_graph_config` 和运行时
compile 标志；`get_server_args()` 在此分支保留原始输入，不能用其
`cuda_graph_config=None` 判断图是否开启。旧提交 `3521625c44` 的检查误读了
该原始字段，已关图仍可能报错，需同步后续修复。新报错会打印生效值；
如仍有某阶段未关闭，检查显式 `--cuda-graph-config` 是否覆盖了阶段开关。

单机 TP8 裁层对照还需将原命令调整为 `--nnodes 1 --node-rank 0
--tp-size 8 --dp-size 1`，使用本机的 `--dist-init-addr`，并让脚本进入单机
启动路径（原脚本按四机 IP 匹配）。模型路径使用已准备好的裁层权重。
DP attention、DP LM head、DeepEP 等选择在两组间保持相同。

## 开启有界采集

在启动服务的 shell 中设置，必须在 Python 进程启动前生效：

```bash
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEBUG_HCCL_DIR=/tmp/k3-hccl/aiv-run1
export SGLANG_DEBUG_HCCL_MAX_STEPS=2
export SGLANG_DEBUG_HCCL_MAX_EVENTS=2000
export SGLANG_DEBUG_HCCL_SAVE_TENSORS=1
export SGLANG_DEBUG_HCCL_MAX_DUMP_MB=512
# 可选：只采指定 K3 层及层内通信，层号使用模型实际 layer_idx。
# export SGLANG_DEBUG_HCCL_LAYERS=0,1,2
# 可选：触发采集后，跳过每类根调用的前 N 次。
# export SGLANG_DEBUG_HCCL_SKIP_STEPS=0
```

然后运行修改后的服务脚本。每个进程只写自己的 `rank-N/events.jsonl`。
目录必须是本次运行的新目录；重复 rank 文件会报错，防止两次运行混写。
多机上使用同一逻辑运行名，每台机器保存自己的全局 rank。

服务完成启动/预热、空闲后，在**每个参与节点**执行：

```bash
touch /tmp/k3-hccl/aiv-run1/START
```

没有 START 文件时不采张量、不消耗步数预算。所有节点触发后才发送请求。
触发后不要再跑额外热身请求或其他客户端负载，以免预算被无关流量消耗。
首次用短文本、单请求、greedy，先确认取证链完整：

```bash
curl --fail --silent --show-error http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  --data-binary @test/manual/ascend/k3_hccl/request.json \
  > /tmp/k3-hccl/aiv-run1/response.json
```

人工结束服务后，改为 `HCCL_OP_EXPANSION_MODE=CCU_SCHED`，
目录改为 `/tmp/k3-hccl/ccu-run1`，重启同一服务、等待预热、在各节点创建
对应 START 文件、发送**相同 request.json**。每次使用新的服务状态。
长上下文故障需随后重放真实固定请求和相同前缀预热步骤；短请求正常不代表修复。

`MAX_STEPS` 按最外层 scope + forward mode 分别计数：K3 prefill 与
DSPARK decode 有独立预算，decode 内 target verify 不重复计数。
`MAX_EVENTS` 是每 rank 总事件数（before 和 after 各计一次）；到上限会写
limit 标记并停止采集。张量保存预算按未压缩 tensor 字节统计，不含文件格式开销；
达到预算后继续写摘要，并以 `dump_skipped` 标记没有保存的张量。
总空间约为每 rank 张量预算乘以 rank 数，外加 JSON/文件格式开销。

## 日志内容和覆盖范围

- `k3.target`：输入 token、position、序列长度、cache 位置、可用 mask、
  logits 和 hidden states；`k3.layer[N]`：层输入/输出、residual、有效残差块数。
- `tp.*`：GroupCoordinator 的 all_reduce、all_gather、all_gather_into_tensor、
  reduce_scatter_tensor、all_to_all_single；记录通信域成员、rank、输入和输出。
  原地 AllReduce 的输入在调用前复制，不会被输出覆盖。
- `deepep.dispatch_a/b`、`deepep.combine_a/b`：路由输入、接收 hidden、
  top-k、权重和 combine 结果。dispatch 顺序、padding 和量化可能变化，
  必须按有效 token/专家映射比较，不可把物理存储 hash 变化直接当成数据损坏。
- `dspark.propose`、`dspark.draft_sample`：draft 输入 hidden/bonus、候选、
  可用的 base/corrected logits、greedy mask 和温度。
- `dspark.accept`：verify candidates/target logits、draft logits、prefix 长度、
  drafts-only `correct_len`、bonus、cap trim、commit 长度和输出 token。
  `dspark.decode`：该轮开始的状态、最终 accept_lens 和输出。
- 摘要涵盖完整逻辑张量：shape/dtype/stride、SHA256、NaN/Inf、min/max/mean/L2。
  小整数张量保存完整值；logits 保存前 8 行的 top-5。启用张量保存后，
  在预算内可获得原 dtype 的完整 CPU `.pt`，用于后续逐元素分析。

没有注入额外 collective、随机数或修改计算张量；默认未设置 DIR 时，
装饰器直接返回原函数，没有每次 forward 的日志分支。
START/JSON/.pt 都是本机文件，不进行上传。

该版本不采集完整 KV/KDA 状态，也不包装所有自定义融合通信内核。
如果层间首次分歧处没有 `tp.*` 记录，需要结合实际调用路径进一步加点，
不能宣称该层没有通信。环境变量只记录为 requested mode；实际 CCU/AICPU
fallback 或通信域覆盖需要另取对应 CANN/HCCL 日志确认。

## 离线比较与下一步

多机日志归拢到对应运行目录，保留各全局 `rank-N`，不要合并 AIV/CCU 两组。
在有源码的机器执行（摘要比较只依赖 Python 标准库）：

```bash
python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-run1 /tmp/k3-hccl/ccu-run1 \
  --output /tmp/k3-hccl/compare.json

# 有 PyTorch 且已保存 .pt 时，增加首个差异的 max_abs/relative_l2 等：
python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-run1 /tmp/k3-hccl/ccu-run1 --tensors \
  --output /tmp/k3-hccl/compare-tensors.json
```

退出码：0 = 已采窗口位级一致；1 = 有差异；2 = 证据缺失/不对齐/预算截断。
0 不证明数值正确，更不排除同步改变后消失的时序问题。
报告只自动定位每 rank 首差，不自动认定责任组件；rank 内的先后顺序
不等于跨 rank 全局时间。逐层比较只能在相同 token 前缀/状态下归因。

判断顺序：所有 rank 同一 collective 的输入是否一致 → 输出是否分歧 →
draft/verify logits 是否翻转候选 → correct_len/commit_lens 是否变化。
输入已不同则继续向前追；候选/判定输入一致但接受结果不同，查接受逻辑、
cap trim、rank 同步与状态。仅 checksum 不同还需量化误差、检查有效区域。

人工启动完成后保留：两组完整服务日志、请求/响应、所有 rank 的诊断目录、
compare.json、版本信息和实际启动命令，再进行下一步定位。
