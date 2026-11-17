# LongCat A3 / 950DT：两层模型固定路由与完整张量逐元素对照

**2026-11-17 A3 实测参考已准备：** 自然路由与固定路由两轮的 416 对完整张量逐元素全部相等。直接使用本次数据在绿区采集，请按 [950DT 启动、重放、数据校验与逐元素对比命令](green_route_20261117/README.md) 执行。代码、493 KiB 路由参考和固定输入通过 Git 获取；完整 A3 张量数据另走 eTrans。下面保留通用四组实验流程。

## 实验范围与代码分支

使用 [hanwlax/sglang 的 `codex/longcat2-route-lock-full` 分支](https://github.com/hanwlax/sglang/tree/codex/longcat2-route-lock-full)。该分支基于原采集版本 `c0e9d9a0cc6b1a37e41677faf0c961fedc668af7`，已接入路由探针和完整张量采集，**不再需要手改源码或从交付目录复制探针**。

配置保持两层、TP8、DP1、attention DP1、EP1，关闭 DeepEP、prefill/decode 图、NEXTN 和 radix cache，使用固定 852-token Control 输入执行无缓存 prefill。这些诊断设置仅用于本次实验，不要应用到生产 EP32 服务。

每轮保存 TP0–7 上现有 52 个探针边界的全部张量元素，预计约 **4.11 GiB/轮**；四轮合计约 **16.44 GiB**。固定路由属于受控实验，不能视为模型精度修复。

分支包含：

| 文件 | 用途 |
|---|---|
| `python/sglang/srt/models/longcat_flash.py` | 已在 identity expert 处理前接入路由捕获和替换 |
| `python/sglang/srt/debug_utils/longcat_activation_trace.py` | 已在记录边界时接入完整张量保存 |
| `python/sglang/srt/debug_utils/longcat_route_lock.py` | 固定路由、完整张量采集模块，默认关闭 |
| `scripts/longcat2_950dt/run_trace.sh` | 两层 TP8 诊断服务启动脚本 |
| `scripts/longcat2_950dt/route_lock_tools.py` | 固定输入重放、A3 路由参考导出、实际路由核验 |
| `scripts/longcat2_950dt/compare_full_route_tensors.py` | 完整张量逐元素绝对误差比较 |
| `scripts/longcat2_950dt/check_route_lock_cpu.py` | 使用合成张量的 CPU 自检，不运行模型、不使用 NPU |

## 1. 两边获取分支并确认源码路径

在各自原容器内执行。下面创建独立 checkout，保留原源码目录中的本地修改。目标目录需要尚不存在；若已经创建过，应先检查其版本和改动，不要直接覆盖。

### A3（蓝区 216）

```bash
git -C /home/hanwlax/workspace/sglang fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git -C /home/hanwlax/workspace/sglang worktree add --detach /home/hanwlax/workspace/sglang-longcat-route-lock FETCH_HEAD
export SGLANG_SOURCE_ROOT=/home/hanwlax/workspace/sglang-longcat-route-lock
git -C "$SGLANG_SOURCE_ROOT" log -1 --format='%H %s'
```

### A5 / 950DT（绿区）

```bash
git -C /mnt/share/hanwlax/workspace/sglang fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git -C /mnt/share/hanwlax/workspace/sglang worktree add --detach /mnt/share/hanwlax/workspace/sglang-longcat-route-lock FETCH_HEAD
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
git -C "$SGLANG_SOURCE_ROOT" log -1 --format='%H %s'
```

两边确认提交号相同。绿区访问 GitHub 沿用现有代理方式；A3 采集数据仍须走蓝区到绿区的既有传输流程。

### 两边检查脚本和探针语法

在各自设置好 `SGLANG_SOURCE_ROOT` 的终端执行：

```bash
python3 -m py_compile \
  "$SGLANG_SOURCE_ROOT/python/sglang/srt/models/longcat_flash.py" \
  "$SGLANG_SOURCE_ROOT/python/sglang/srt/debug_utils/longcat_activation_trace.py" \
  "$SGLANG_SOURCE_ROOT/python/sglang/srt/debug_utils/longcat_route_lock.py" \
  "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" \
  "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/compare_full_route_tensors.py"
bash -n "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh"
```

可在 A3 原容器执行 CPU 自检：

```bash
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/check_route_lock_cpu.py" --control /tmp/longcat-control.json
```

自检读取 Control 的输入 token，但计算使用合成张量。它验证探针及比较工具，不代替 A3/950DT 实机采集。

后续直接执行分支内的启动脚本和工具，不再使用进度目录下旧的 `run_trace.sh` 或 `route_lock_20261117/` 副本。启动日志中的 `source=` 应指向新 checkout。模型目录在命令中显式指定；如实际目录不同，仅调整 `MODEL_PATH`。默认 HTTP 端口为 10086，如修改端口，重放命令需同时传入对应的 `--url`。

## 2. A3 自然路由采集与参考文件导出

使用两个终端：一个启动服务，另一个在服务就绪后重放。

每轮使用未用过的运行目录；如果文档中的目录名已存在采集文件，统一改用一组新名称。**每个服务实例及其运行目录只发送一次匹配的 Control 请求**；不要同时运行包含相同 Control 输入的评测或健康检查请求。

### A3 启动终端

```bash
cd /home/hanwlax/workspace/progress/longcat2.0
export SGLANG_SOURCE_ROOT=/home/hanwlax/workspace/sglang-longcat-route-lock
export MODEL_PATH=/home/l00951279/longcat2/LongCat-2.0-INT8-part
run="$PWD/route_runs/A3_natural"
mkdir -p "$run"
set -o pipefail
export SGLANG_DEBUG_LONGCAT_ACTIVATIONS=1
export LC_ROUTE_MODE=capture
export LC_ROUTE_OUT="$run/routes"
export LC_FULL_DUMP_OUT="$run/full"
unset LC_ROUTE_REFERENCE
HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable bash "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh" A3 2>&1 | tee "$run/serve.log"
```

### A3 重放终端

服务就绪后执行：

```bash
cd /home/hanwlax/workspace/progress/longcat2.0
export SGLANG_SOURCE_ROOT=/home/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" replay \
  --control /tmp/longcat-control.json \
  --log "$PWD/route_runs/A3_natural/serve.log" \
  --out "$PWD/route_runs/A3_natural/client" --require-full
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" pack \
  --routes "$PWD/route_runs/A3_natural/routes" \
  --out "$PWD/route_runs/A3_reference.json"
```

预期输出：

- `REPLAY_OK`：保存了 **416 个完整张量、16 条路由记录**。
- `PACK_OK`：成功导出路由参考文件。

参考文件包含两层、全部 852 个 token 行、每行全部 12 个专家 ID 及缩放后的路由权重；保存位置位于 identity expert 处理之前。如果不同 TP rank 的路由数据不一致，导出工具会报错。

旧 JSONL 只保存三个 token 行的抽样信息，无法用它重建完整路由参考。

## 3. A3 固定路由对照

正常停止上一轮 A3 诊断服务，再启动固定路由服务。以下命令显式设置本轮工作目录和源码路径。

### A3 启动终端

```bash
cd /home/hanwlax/workspace/progress/longcat2.0
export SGLANG_SOURCE_ROOT=/home/hanwlax/workspace/sglang-longcat-route-lock
export MODEL_PATH=/home/l00951279/longcat2/LongCat-2.0-INT8-part
set -o pipefail
run="$PWD/route_runs/A3_fixed"
mkdir -p "$run"
export LC_ROUTE_MODE=ids_weights
export LC_ROUTE_REFERENCE="$PWD/route_runs/A3_reference.json"
export LC_ROUTE_OUT="$run/routes"
export LC_FULL_DUMP_OUT="$run/full"
HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable bash "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh" A3 2>&1 | tee "$run/serve.log"
```

### A3 重放终端

```bash
cd /home/hanwlax/workspace/progress/longcat2.0
export SGLANG_SOURCE_ROOT=/home/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" replay \
  --control /tmp/longcat-control.json \
  --log "$PWD/route_runs/A3_fixed/serve.log" \
  --out "$PWD/route_runs/A3_fixed/client" --require-full
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" audit \
  --routes "$PWD/route_runs/A3_fixed/routes" \
  --reference "$PWD/route_runs/A3_reference.json"
```

预期出现 `AUDIT_OK`，表示实际采用的完整路由张量与参考一致。

## 4. 从蓝区向绿区传入数据

源码由两边检出同一分支获得。按已有的 **eTrans → 绿区 PC → 绿区主机** 流程传入以下 A3 数据：

```text
route_runs/A3_reference.json
route_runs/A3_natural/routes/
route_runs/A3_natural/full/
route_runs/A3_fixed/routes/
route_runs/A3_fixed/full/
```

放在 A5 进度目录下，保留上述目录结构。两轮 A3 `.pt` 张量数据合计约 **8.22 GiB**。

绿区文件保留在绿区，只回传文字摘要。

## 5. A5 自然路由采集

A5 使用第 1 节检出的同一分支，保持原来的两层 `run_trace` 配置。

### A5 启动终端

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
export MODEL_PATH=/mnt/share/weight/LongCat-2.0-FP8-cut
run="$PWD/route_runs/A5_natural"
mkdir -p "$run"
set -o pipefail
export SGLANG_DEBUG_LONGCAT_ACTIVATIONS=1
export LC_ROUTE_MODE=capture
export LC_ROUTE_OUT="$run/routes"
export LC_FULL_DUMP_OUT="$run/full"
unset LC_ROUTE_REFERENCE
HCCL_OP_EXPANSION_MODE=AI_CPU bash "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh" A5 2>&1 | tee "$run/serve.log"
```

### A5 重放终端

服务就绪后执行：

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" replay \
  --control "$PWD/longcat_completion_1_kr12o5/Control.json" \
  --log "$PWD/route_runs/A5_natural/serve.log" \
  --out "$PWD/route_runs/A5_natural/client" --require-full
```

## 6. A5 固定路由采集

正常停止上一轮 A5 诊断服务，再启动固定路由服务。以下命令显式设置本轮工作目录和源码路径。

### A5 启动终端

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
export MODEL_PATH=/mnt/share/weight/LongCat-2.0-FP8-cut
set -o pipefail
run="$PWD/route_runs/A5_fixed"
mkdir -p "$run"
export LC_ROUTE_MODE=ids_weights
export LC_ROUTE_REFERENCE="$PWD/route_runs/A3_reference.json"
export LC_ROUTE_OUT="$run/routes"
export LC_FULL_DUMP_OUT="$run/full"
HCCL_OP_EXPANSION_MODE=AI_CPU bash "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/run_trace.sh" A5 2>&1 | tee "$run/serve.log"
```

### A5 重放终端

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" replay \
  --control "$PWD/longcat_completion_1_kr12o5/Control.json" \
  --log "$PWD/route_runs/A5_fixed/serve.log" \
  --out "$PWD/route_runs/A5_fixed/client" --require-full
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/route_lock_tools.py" audit \
  --routes "$PWD/route_runs/A5_fixed/routes" \
  --reference "$PWD/route_runs/A3_reference.json"
```

预期出现 `AUDIT_OK`。只有核验通过，才能把这轮结果当作固定路由对照。

重放使用 `max_new_tokens=1`，只执行无缓存 prefill，不将生成的 token 继续送入 decode，因此生成结果分叉不会改变后续输入。

无需指定客户端 RID：诊断功能通过完整固定输入 token 和位置的哈希匹配来启用。

## 7. 在绿区逐元素比较完整张量

使用 CPU PyTorch 执行，无需 NPU。继续在 A5 进度目录下操作。

### 比较自然路由组和固定路由组

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/compare_full_route_tensors.py" \
  --a3 "$PWD/route_runs/A3_natural" --a5 "$PWD/route_runs/A5_natural" \
  --out "$PWD/route_runs/compare_natural" > "$PWD/route_runs/compare_natural.log"
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/compare_full_route_tensors.py" \
  --a3 "$PWD/route_runs/A3_fixed" --a5 "$PWD/route_runs/A5_fixed" \
  --require-fixed --out "$PWD/route_runs/compare_fixed" > "$PWD/route_runs/compare_fixed.log"
```

### 检查 A3 自身的重放一致性

下面命令的第二个输入虽然使用参数名 `--a5`，实际传入的是 **A3 固定路由采集结果**，用于比较 A3 自然组与固定组：

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
python3 "$SGLANG_SOURCE_ROOT/scripts/longcat2_950dt/compare_full_route_tensors.py" \
  --a3 "$PWD/route_runs/A3_natural" --a5 "$PWD/route_runs/A3_fixed" \
  --out "$PWD/route_runs/compare_A3_self" > "$PWD/route_runs/compare_A3_self.log"
```

查看固定路由组摘要：

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang-longcat-route-lock
cat "$PWD/route_runs/compare_fixed/summary.txt"
```

### 输出内容

| 文件 | 内容 |
|---|---|
| `summary.txt`、`summary.json` | 按边界汇总所有 rank 的计数、首个差异和最大绝对差异，包含 TP rank、token、列坐标及两边原值 |
| `per_tensor.jsonl` | 每个 rank 的完整张量比较结果，不抽样 |

误差按以下区间计数：

| 区间 | 含义 |
|---|---|
| `exact` | 完全相等 |
| `(0,0.0001)` | 非零绝对误差小于 0.0001 |
| `[0.0001,0.001)` | 绝对误差为 0.000x 量级 |
| `[0.001,0.01)` | 绝对误差为 0.00x 量级 |
| `[0.01,0.1)` | 绝对误差为 0.0x 量级 |
| `>=0.1` | 绝对误差不小于 0.1 |

NaN/Inf 单独计数。整数 ID 只做精确相等比较，不按浮点误差分桶。

比较器在 CPU 上分块处理，但会检查**所有元素**。跨 rank 复制的张量在每个 rank 上各计一次，因此汇总计数不是去重后的逻辑激活元素数量。不计算 L2，也不设置精度通过或失败的容差阈值。

## 8. 如何解释结果

- 如果 A3 自然组与固定组出现明显变化，应先排查重复运行的一致性和采集影响，再把跨平台差异归因于路由。
- 固定专家 ID 和路由权重后，如果 A3/A5 差异缩小，说明路由选择和/或加权系数参与了已有差异的形成；这不等于证明 TopK 实现错误。
- 剩余差异仍可能来自不同的 MoE 输入、INT8/FP8 专家参数、量化和实际计算。**专家 ID 相同不代表专家权重张量相同。**
- 后续可选做 `ids` 模式实验：只固定专家 ID，重新计算各个被选专家对应的路由系数，不会把原槽位的权重直接配给新专家。首轮上述四组实验不需要此模式。
- 完整张量的同步拷贝会改变时序。本轮用于分析计算差异，不能用于判断生产吞吐，也不能证明未插桩服务不存在竞态。

## 9. 诊断结束后关闭开关

重启服务前执行：

```bash
unset LC_ROUTE_MODE LC_ROUTE_OUT LC_ROUTE_REFERENCE LC_FULL_DUMP_OUT
```

诊断模块默认关闭。不要把固定路由结果用作 GSM8K 精度验收。
