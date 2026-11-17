# A5 完整 FP8：与 A3 对齐的 Eliza 全张量采集

源码分支：`hanwlax/sglang:codex/longcat2-a5-full-fp8-capture`。
本目录中的生成器、helper、fixture、启动/重放和核验工具一并版本化。
推理实现冻结于 `c0e9d9a0cc6b1a37e41677faf0c961fedc668af7`；不会自动带入
绿区现有 checkout 的手工探针或未提交模型改动。修改 helper 后必须生成新 runtime、
重启四机；仅修改外层文件不会改变已运行服务。

本配置：完整38 block，TP32 / attention DP2 / attention TP16 / EP32 / MoE TP1，
DeepEP auto、AI_CPU，关 prefill/decode 图、关 NEXTN、关 radix cache。
`MOE_ENABLE_TOPK_NEG_ONE=1`、`DEEPEP_HYBRID_DEPLOYMENT=1`、
`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`，MLAPO=0、ARCH35_REQUANT_BLOCK_FP8=0。
保留真实 router/TopK/路由权重/dispatch/combine/identity/残差计算；仅固定 token
前缀以对齐下一步输入，不固定专家，不替换算子。

## 1. 获取源码，准备独立 runtime

以下在绿区原容器中运行。使用独立 worktree 保留原 `run.sh` 和现有源码修改。
若四机共享这些路径，只需获取源码和生成 runtime **一次**，其余节点核对 receipt；
若不共享，则四机分别执行，保持相同源码提交、fixture 和配置。

```bash
REPO=/mnt/share/hanwlax/workspace/sglang
export SRC=/mnt/share/hanwlax/workspace/sglang-a5-eliza-full
git -C "$REPO" fetch https://github.com/hanwlax/sglang.git codex/longcat2-a5-full-fp8-capture
git -C "$REPO" worktree add --detach "$SRC" FETCH_HEAD
git -C "$SRC" rev-parse HEAD

export TOOLS="$SRC/test/manual/ascend/longcat2_baseline"
export CAPTURE_ROOT=/mnt/share/hanwlax/workspace/progress/longcat2_0/a5_eliza_full
export RUNTIME="$CAPTURE_ROOT/runtime_eliza_v1"
export MODEL_PATH=/mnt/share/weight/LongCat-2.0-FP8
mkdir -p "$CAPTURE_ROOT"
cd "$TOOLS"
python3 -c 'import torch, torch_npu, blake3; print(torch.__version__)'
bash capture_eliza_a5.sh prepare "$SRC"
```

如果仅缺 blake3，执行 `python3 -m pip install 'blake3>=1.0.0'` 后再准备。
`prepare` 等价于生成器的 `--trace-topology a5-dp2 --full-boundary-layers 0..37`
加 `--with-internal-attention --load-barrier-timeout 3600`，随后检查 runtime 哈希
并运行 CPU 完整行映射测试。应出现 `CAPTURE_RUNTIME_OK` 和
`FULL_BOUNDARY_CPU_PASS`。OUT 已存在会拒绝覆盖；修改后用 `runtime_eliza_v2` 等新目录。

每台容器都要设置相同的 `SRC/TOOLS/CAPTURE_ROOT/RUNTIME/MODEL_PATH` 并核对：

```bash
cd "$TOOLS"
python3 check_capture_runtime.py --runtime "$RUNTIME" --topology a5-dp2
df -h "$CAPTURE_ROOT"
```

四机 `receipt_sha256` 应一致。A3 本次采集约406 GiB；建议共享数据盘预留至少600 GiB。
非共享盘中，主要张量落在 .105/.113（DP0 的 global TP0..15），建议各预留至少300 GiB，
其余两机也需保留日志/receipt空间。该容量是按 A3 估计，A5 实际大小可能不同。

## 2. 四机启动

停止旧的自有模型服务后启动。此脚本不主动 kill 其他进程。
四机使用同一个**新** RUN 名称；重试使用 `_02`，不混用旧文件。

```bash
export RUN=a5_eliza_full_01
cd "$TOOLS"
# 四机分别设置 NODE_RANK：.105=0、.113=1、.121=2、.122=3
NODE_RANK=0
bash capture_eliza_a5.sh start "$NODE_RANK" "$RUN" --dry-run
bash capture_eliza_a5.sh start "$NODE_RANK" "$RUN"
```

节点与内部采集范围：

| 主机 | node rank | global TP | DP0 内部张量 |
| --- | --- | --- | --- |
| 141.61.45.105 | 0 | 0–7 | 采集 |
| 141.61.45.113 | 1 | 8–15 | 采集 |
| 141.61.45.121 | 2 | 16–23 | 不采，仍执行真实 EP 通信/专家计算 |
| 141.61.45.122 | 3 | 24–31 | 不采，仍执行真实 EP 通信/专家计算 |

内部采集固定 `LC_INTERNAL_LAYERS=0,7,37`、`LC_INTERNAL_RANKS=0..15`。
每个 block 的两个 Attention 都采，attention_id 为 `0/1、14/15、74/75`。
HTTP 在 .105:10086，rendezvous .105:5017。默认 HCCL 网卡 `enp34s0f1`，
Gloo `data0.3001`；若容器实际接口不同，在四机启动前设置对应变量。
启动日志会核验完整38层FP8 checkpoint，并记录实际环境、包版本和模型元数据哈希。

## 3. 重放：只在一个客户端执行

四机就绪后，在 .105 原容器的新终端设置同样路径及 RUN，再执行：

```bash
cd "$TOOLS"
export BASE_URL=http://141.61.45.105:10086
bash capture_eliza_a5.sh replay "$RUN"
```

客户端先读取有效 server 配置，再串行发送六个请求：

| prefix length | cold prefill | 固定首输出 | 下一次真实 decode position | 重复 |
| --- | --- | --- | --- | --- |
| 1 | 853 tokens | 78261 | 853 | r0预热，r1/r2正式 |
| 11 | 863 tokens | 444 | 863 | r0预热，r1/r2正式 |

直接提交 A3 `fixtures/eliza_853.json` 中的 IDs，不重新 tokenize。
prompt SHA256：`1b8d40fb3f004235b35182563c4883cb4ef8ce14bf6f7ff55484afe8e262196c`。
11-token 前缀是 ` Eliza earns 40 * 10 = `，是指定的算术对照前缀，不是从原 A5
故障回答恢复的 token IDs。两边均发送完整固定前缀，因此不会因自然生成分叉而比较不同输入。
每个请求 `max_new_tokens=2`、temperature=0、ignore_eos=true、timeout=3000。
第二个输出由真实模型计算；不要求 A5 匹配 A3 的答案或 next token。
不要同时运行 GSM8K 或其他请求。超时/异常会停止后续提交，先查看服务首错，再决定新一轮。

## 4. 核验和抓日志

共享盘：完成六个请求后，运行一次完整校验（仅 CPU，不需要继续占用 NPU）：

```bash
cd "$TOOLS"
bash capture_eliza_a5.sh verify "$RUN"
grep -hE 'BASELINE_CONFIG|LONGCAT_FULL_BOUNDARY|LONGCAT_FIXED_V4_FORCE|Scheduler hit|RuntimeError|Traceback' \
  "$CAPTURE_ROOT/results/$RUN/dumps/serve_rank"*.log \
  > "$CAPTURE_ROOT/results/$RUN/important.log"
grep -E '^REPEAT |^CAPTURE_SUMMARY ' "$CAPTURE_ROOT/results/$RUN/verification.log"
```

预期：32 worker receipts、128个完成索引、2304个内部文件；boundary文件数由实际有效行计算，
不硬编码要求等于 A3。核验会读取所有有效 boundary payload，验证SHA、shape、行/位置映射、
NaN/Inf，同时逐个反序列化内部文件。`REPEAT` 给出相同平台两次采集的不等张量数和首个差异位置；
所有差异文件名在 `verification.json` 中。重复不等本身不会自动判精度失败；非有限有效边界返回 exit2。
没有 `.json` 完成索引的中断数据不能充作完整采集。内部 raw cache 的未使用槽不能直接算作模型激活。

非共享盘：四个节点分别有 `results/$RUN/dumps`。在绿区内复制到核验机的四个独立父目录，
同时保留各节点的 `launch_rankN.json` 和 `runtime_rankN.json`。例如：

```text
/green/gather/node0/{dumps/,launch_rank0.json,runtime_rank0.json}
/green/gather/node1/{dumps/,launch_rank1.json,runtime_rank1.json}
/green/gather/node2/{dumps/,launch_rank2.json,runtime_rank2.json}
/green/gather/node3/{dumps/,launch_rank3.json,runtime_rank3.json}
```

不要将一个不完整节点目录当成完整数据，也不要覆盖同名文件。然后在核验机：

```bash
python3 "$TOOLS/verify_full_capture.py" --topology a5-dp2 \
  --manifest "$CAPTURE_ROOT/results/$RUN/client_eliza/manifest.json" \
  --dump /green/gather/node{0,1,2,3}/dumps \
  --verify-payloads --workers 8 --out "$CAPTURE_ROOT/results/$RUN/verification.json" \
  | tee "$CAPTURE_ROOT/results/$RUN/verification.log"
```

## 5. 后续比较需要保留的文件

- `client_eliza/`：fixture哈希、每次输入/响应、manifest；`server_info.json`。
- 四机 `launch_rank*.json`、`runtime_rank*.json` 和 `dumps/serve_rank*.log`。
- `dumps/worker.tp*.json`：32-rank实际拓扑；`lcfix2-*.json/.pt`：完整索引。
- `dumps/lcfix2-*.full/*.pt`：全38层完整有效边界元素；`dumps/internal/*.pt`：0/7/37内部数据。
- `verification.json/.log`：完整性、有限值、重复一致性摘要和首差文件位置。

A3 对应归档在蓝区217：
`/home/hanwlax/workspace/progress/longcat2.0/a3_window_20261117_5cf81e6/archives/a3_capture_dp0_20261117_2044/`。
A3自然生成 GSM8K 为183/200（0.915），上述两个前缀46386对边界哈希完全一致；
A5这套工具尚需绿区实际启动和采集验证。

真正的跨平台比较须先对齐 token/position/全局head/全局专家ID。
router logits、选中集合和权重分别比较；路由已变时，不把MoE输出差直接当同输入专家算错。
`layer.output` 为待完成的分支贡献，`layer.residual` 是独立残差；dense FFN 本地分片因TP64/32
不同不能直接逐元素硬比。绝对误差单位是原激活数值，relative L2无量纲；不设置未经验证的统一阈值。
本工具不做 A3↔A5 跨量化数值判定；数据落盘后再定位最早明显差异。
绿区张量留在绿区，先回传 `REPEAT`/`CAPTURE_SUMMARY` 和必要日志文字。

## 工具验证记录

本分支生成的 A5 DP2 runtime 已通过 CPU 行映射/填充/快照测试；四机启动命令通过
合成38层FP8配置的 dry-run，并检查了错误拓扑、开图/NEXTN、裁层、内部rank缺失的拒绝。
新核验器完整读取过上述真实 A3 数据：92772个边界文件、2304个内部文件通过，
边界NaN/Inf为0，46386对重复边界完全一致。CPU/dry-run及A3数据读取不代表A5 NPU已验证。
