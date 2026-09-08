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
- 当前裁层 target 与原 draft 不匹配，两种通信模式下接受长度均为 1；
  当前运行还使用环境变量模拟接受长度。因此此阶段只定位数值边界和路由
  变化，不以接受长度判断通信精度。真实接受长度的因果验证留给完整模型。

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

## 首差出现在 AllReduce 输出时

若所有 rank 的首差都是同一 AllReduce 的 after 事件，可直接使用已保存的
输入/输出构造 CPU FP64 SUM 参考。例如 after 事件号为 3：

```bash
python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-run1 /tmp/k3-hccl/ccu-run1 \
  --tensors --allreduce-event 3 \
  --output /tmp/k3-hccl/compare-reference.json
```

工具从参考 rank（默认 0，可用 `--reference-rank` 指定）的通信域读取全部
成员，验证 `.pt` 与日志摘要匹配、逐 rank 两组输入位级相同，再按域内 rank
顺序进行 FP64 求和。每种模式输出报告包含相对 FP64 的误差、与“FP64 求和
后仅一次转回 BF16/原 dtype”结果不同的元素数，以及域内输出是否一致。
FP64 是高精度参考，不承诺所有输入下都等于精确实数求和。此功能只适用于
本诊断 `tp.all_reduce` 的 SUM；需要该次所有成员输入/输出的 `.pt`。
缺失文件、输入不一致或摘要校验失败会写 `allreduce_reference_error`。

该参考分析独立于后续事件是否对齐：例如 event 15 停止对齐，不影响验证
完整的 event 2/3。新报告会在 `alignment_mismatch` 中给出停止位置两侧的
scope 和 metadata，可区分调用路径变化、路由计数变化等；仍不跳过这些差异
强行比较后续事件。`bitwise_different_tensors` 只统计停止对齐之前的窗口。
hidden tensor 的 `last_dim_argmax_different` 是通道索引变化，不是输出 token
变化；接受长度因果关系仍需关联到真实 draft/target logits 和接受结果。

## 下一轮：先查裁层路由，再用完整模型验证接受长度

已采数据确认第 0 层 AllReduce 的所有 rank 输入在 AIV/CCU 两组相同，
输出不同，且各模式域内输出一致。两组相对 FP64 的 L2 误差接近，
不能只凭此认定 CCU 精度显著更差。后续第 1 层 dispatch 接收专家计数不同，
下一步先查看发送侧 `deepep.dispatch_a` 的 `topk_output.topk_ids` 和
`topk_output.topk_weights`，判断路由是否在 dispatch 前已经分歧。

**现有数据无需重新采集**，更新分析脚本即可使用简短输出：

```bash
python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-run1 /tmp/k3-hccl/ccu-run1 \
  --tensors --allreduce-event 3 --brief \
  --output /tmp/k3-hccl/compare-reference.json

python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-run1 /tmp/k3-hccl/ccu-run1 \
  --inspect-scope deepep.dispatch_a --inspect-rank 0 --inspect-limit 8 \
  --brief --output /tmp/k3-hccl/inspect-routing-rank0.json
```

`--brief` 合并相同的 rank 摘要；归约参考在验证所有成员后才省略相同输出。
`--output` 始终保留完整比较报告。scope 检视默认仅展示指定 rank 的前 16 个
匹配事件，`omitted_events` 表示未展示数；需要时增加 `--inspect-limit`。
rank 0 用于先看位置，归因需要检查全部参与 rank，可将 `--inspect-rank`
依次改为 0 至 7，并使用不同输出文件。

检视功能独立读取两侧记录，event 15 后仍能查看，但**不进行自动对齐**。
相同 root 序号不保证相同前缀、cache 状态或有效行。发送侧 top-k 已不同则
接收计数不同可能是路由的后果；若所有发送侧路由相同、接收计数仍不同，
再检查 DeepEP 映射、padding 和分发。缺少 `.pt` 时 hash 可提示差异，
不能量化变更的 token 数或专家集合。`saved_file_exists` 只检查文件存在，
完整归约参考仍会校验 `.pt` 内容与 JSON 摘要。

裁层对照可以保留相同的模拟接受设置以控制执行形状；重新采集时两侧设置
必须相同。模拟长度相同不保证后续 token、路由或状态相同。建议补做独立
重启的 AIV/AIV 和 CCU/CCU 对照，先确认首个归约差异是否具有可重复性。

完整模型阶段才在**所有节点的最终启动环境**关闭接受和专家路由模拟：

```bash
export SGLANG_SIMULATE_ACC_LEN=-1
unset SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
export SGLANG_SIMULATE_UNIFORM_EXPERTS=0
export SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=0
```

检查启动脚本没有再次覆盖；保留完整 target、匹配 draft、原问题拓扑。
仍按「启动 → 清缓存 → 填充相同前缀 → START → 正式缓存测试」执行，
第一轮单请求、固定输入、temperature=0，两个模式分别使用新目录和新服务。
不要在填充前缀后再次 flush cache。先确认实际 cache 命中和有效输入一致。

新日志 manifest 记录 `simulation`；`dspark.accept` 的 before metadata
还记录执行器实际持有的 `self._simulate_acc_len`。新增 eager scope
`dspark.accept_raw` 在模拟覆盖和 TP 同步前记录真实内核返回值：
`result.0` = drafts-only correct_len，`result.1` = bonus，
`result.2` = cap_trim_lens。外层 `dspark.accept` 的 after 记录最终结果。
没有修改接受算法或模拟设置。raw 结果仍受当前模型及历史状态影响，
不能把模拟轨迹中的 raw 长度作为完整模型真实运行的接受率。

完整模型采集后使用：

```bash
python3 scripts/compare_hccl_debug.py \
  /tmp/k3-hccl/aiv-full-run1 /tmp/k3-hccl/ccu-full-run1 \
  --inspect-scope dspark.accept --inspect-rank 0 --inspect-limit 8 \
  --brief --output /tmp/k3-hccl/inspect-accept-rank0.json
```

scope 匹配包含嵌套的 accept_raw。先关联首轮候选、target logits、raw
correct_len、最终 commit_lens；前缀已分歧后不继续按轮号逐元素归因。
旧日志无 simulation 时标记 unknown，无 accept_raw 时无法追溯覆盖前结果。
如果 `matching_events=0`，检查事件预算、是否实际进入 DSPARK decode，
以及是否更新了采集端源码；若 `.pt` 被 prefill 耗尽预算，按磁盘空间调整
MAX_DUMP_MB 或缩小层采集范围后重新采集，不从摘要推断完整 logits。

## 独立重放已保存的 AllReduce（单机 TP8）

使用 `scripts/replay_hccl_allreduce.py`，与 `compare_hccl_debug.py` 放在同一
目录。脚本不导入 SGLang，不加载模型，不需要 START、请求或前缀填充。
`check` 和 `summarize` 只需要 CPU PyTorch；`run` 需要 torch_npu 和 NPU。
此版本只支持单机完整通信域 `[0, ..., N-1]`、原始连续输入、SUM，
正好覆盖当前 event 3。其他子通信域、跨机或非连续输入会拒绝，而非静默
重映射。它会验证双方所有 rank 的输入相同、输入输出 `.pt` 与摘要一致。

先人工停止原模型服务，使用同一台服务器、同一容器/软件版本、同一组物理
NPU 和原 rank 到设备映射。保持采集时的 HCCL_BUFFSIZE、HCCL_ALGO、确定性
配置和网卡设置；不要同时调整这些参数。确保 torchrun 的全部子进程都能
看到原 8 张卡，而非继承某个 SGLang worker 的单卡可见配置。
保存 CANN/HCCL、驱动、固件版本以及实际环境；脚本另外记录 torch、torch_npu、
设备名称、请求的展开模式和部分通信环境变量。实际引擎仍需 HCCL 日志确认。

在源码根目录运行预检（不会启动通信）：

```bash
python3 scripts/replay_hccl_allreduce.py check \
  --left /tmp/k3-hccl/aiv-run1 \
  --right /tmp/k3-hccl/ccu-run1 --event 3
```

期望 `ready=true`、group_ranks 为 0–7、shape 为 `[1024, 7168]`、dtype 为
`torch.bfloat16`。失败则先解决缺失文件、输入不一致或布局不支持，不继续运行。
首次每轮读取 CPU 输入并做显式设备同步；每个 rank 只有自己的独占工作 buffer，
直接调用 `torch.distributed.all_reduce(..., SUM)`，归约后同步再拷回 CPU。
每轮都从原输入恢复，包括 warmup，绝不把上一次归约输出作为下一次输入。
2 次 warmup 也保留证据，随后执行 10 次测量；这里“测量”指正确性观察，
不测延迟或吞吐。每一种不同输出保存一份 `.pt`，每轮记录 hash 和 FP64 误差。

分别用全新的进程运行两种模式，顺序执行（以下 Bash 块）：

```bash
set -o pipefail
export OMP_NUM_THREADS=1
for mode in AIV CCU_SCHED; do
  out="/tmp/k3-hccl/replay-${mode}-run1"
  mkdir -p "$out" || break
  HCCL_OP_EXPANSION_MODE="$mode" \
  python3 -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/replay_hccl_allreduce.py run \
    --left /tmp/k3-hccl/aiv-run1 \
    --right /tmp/k3-hccl/ccu-run1 --event 3 \
    --mode "$mode" --warmup 2 --iterations 10 \
    --output "$out" 2>&1 | tee "$out/run.log"
  replay_status=$?
  if [ "$replay_status" -ne 0 ]; then
    echo "Replay failed: $mode (exit $replay_status); inspect $out/run.log"
    break
  fi
done
```

两次重放均从同一份 left 输入加载，right 输入只参与相等性校验和原模型结果
对照。输出目录必须是新目录，已有任意 `rank-N` 子目录会报错，避免混入旧结果；
失败后再次运行也需改用新目录。目录只保存本次重放证据，不修改原始采集文件。
如果启动通信失败，查最早的 rank/主进程异常，不把其他 rank 的 TCPStore
断连报错直接判定为 AllReduce 数值错误。

两组成功后汇总：

```bash
python3 scripts/replay_hccl_allreduce.py summarize \
  /tmp/k3-hccl/replay-AIV-run1 \
  /tmp/k3-hccl/replay-CCU_SCHED-run1 \
  --output /tmp/k3-hccl/replay-summary.json
```

汇总验证所有 rank 完成、迭代数一致、已保存输出的 shape/dtype/hash 正确，
并检查两组重放使用同一组输入。屏幕上仅在域内一致时省略重复 rank 的首轮
结果，完整 JSON 保留各 rank 首轮结果，每轮详情在 `rank-N/report.json`。

重点字段：

- `all_ranks_repeatable_including_warmup`：每个 rank 在全部 12 轮是否位级稳定。
- `all_ranks_equal_each_iteration`：每一轮所有 rank 的输出是否位级一致。
- `all_outputs_match_model.left/right`：所有轮次、所有 rank 是否分别等于
  原 aiv-run1 / ccu-run1 保存的模型内归约输出；left/right 指来源目录。
- `any_nonfinite`：任意输出是否出现 NaN/Inf。
- `max_abs_vs_fp64`、`relative_l2_vs_fp64`：相对全部原输入 CPU FP64 SUM 的误差。

如果 AIV 重放稳定匹配 left、CCU 重放稳定匹配 right，且两组 hash 不同，
则强力支持该次首差可脱离模型算子、专家路由和模型调度复现。仍需根据归约
精度要求区分允许的浮点差异与实现错误，不能直接证明完整模型接受长度根因。
若无法匹配原模型输出，不立即归因于模型调度：新通信域、内存地址、算法选择、
通信配置或实际引擎也可能不同，先核对这些条件。进程内重复稳定后，可以用
run2 新目录再启动一轮，检查跨进程重启的重复性。
