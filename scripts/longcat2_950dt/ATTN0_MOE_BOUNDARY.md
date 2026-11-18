# L0 attn0.local_output → moe.input：已有 dump 的 CPU 对照

无需启动模型、无需 NPU、无需重复采集 Attention。代码位于 `codex/longcat2-route-lock-full` 分支，入口为 `scripts/longcat2_950dt/check_attn0_moe_boundary.py`。

本工具限定本次两层诊断的 **L0、TP8、DP1/attention DP1、EP1、852-token 无缓存 prefill**。不用于生产 TP32/EP32 或 decode。

## 检查什么

源代码在 `longcat_flash.py` 中先记录 `attn0.local_output`，再执行 `attn_boundary.finish()` 和 `moe_boundary.prepare()`，记录 `moe.input`。第一个 dense FFN 从 MoE 边界复用输入，同时记录 `ffn0.residual_in`。

因此现有 dump 已含以下 5 个边界、全部 8 个 rank，共 40 个完整张量：

| 边界 | 用途 |
|---|---|
| `embedding.output` | L0 起始残差 R0 |
| `attn0.local_output` | TP0–7 的 Attention 本地贡献 |
| `ffn0.residual_in` | Attention 归约、残差相加后保存的 BF16 残差 R1 |
| `moe.input` | 归一化后的实际输出 |
| `ffn0.input` | 检查它是否确实复用了 `moe.input` |

工具先核验输入 ID、positions、路由元数据、文件头、形状、dtype、完整行映射。检查复制张量在 TP0–7 是否一致，以及 `moe.input == ffn0.input`。只有这些检查通过，才用 TP0 代表复制值，并汇总全部 8 个 rank 的本地贡献。发现 rank 差异、缺失文件、混合输入或非有限输入时会停止，不会静默继续使用 TP0。

随后对**全部 852×8192 个元素**报告绝对差分桶、首个差异和最大差异坐标，不抽样、不计算 L2。没有自动精度通过/失败阈值。

## 必须区分的舍入和证据范围

当前 dump **没有独立保存 HCCL 归约后的张量 S**，也没有保存运行时 norm 权重。工具从对应 checkpoint 读取 `model.layers.0.post_attention_layernorm.0.weight` 与 `rms_norm_eps`，按采集配置假设 norm 参数运行时为 BF16；这不等于已经验证运行时参数加载一致。

实际路径可表示为：

```text
8 个 local_output → HCCL 归约 S
S + embedding.output → 融合 add_rms_norm
                         ├─ BF16 residual_out = ffn0.residual_in
                         └─ norm output = moe.input
```

`npu_add_rms_norm` 可以使用相加后尚未舍入的值进行归一化。因此，直接对 BF16 `ffn0.residual_in` 做 RMSNorm，不一定与融合输出逐位相同。

工具明确报告以下检查，不挑选“最接近”的结果冒充真实 HCCL 算法：

- `norm_from_stored_residual`：实际保存的 BF16 R1 → CPU RMSNorm，分别用 FP32、FP64 算术；仅是存储值参考。
- `reduce_plus_residual`：候选归约结果 + R0 → BF16，与实际 R1 对比。
- `fused_add_norm`：候选归约结果 + R0，在加法结果未额外舍入到 BF16 时归一化，分别用 FP32、FP64 算术，与实际 `moe.input` 对比。
- 归约候选为 FP64 累加后一次 BF16 舍入、BF16 正序串行、BF16 逆序串行、BF16 配对树。**这些只是候选模型，不穷尽实际 HCCL 的归约过程，也不构成误差上下界。**

FP64 参考输出采用直接 BF16 最近偶数舍入，避免经 FP32 中转的二次舍入。

## 1. 绿区更新独立源码

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
git status --short
git fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git merge --ff-only FETCH_HEAD
```

有本地修改时不要强制覆盖。只需 CPU PyTorch；使用 `--model` 读取权重还需 safetensors，原服务容器已有这些依赖。脚本不导入 SGLang 服务或 torch_npu。

## 2. 先检查 950DT 自己的边界，不依赖 A3 大包

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
set -o pipefail
python3 scripts/longcat2_950dt/check_attn0_moe_boundary.py \
  --run /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/A5_fixed_20261117 \
  --model /mnt/share/weight/LongCat-2.0-FP8-cut \
  --out /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/boundary_A5_20261118 \
  --threads 4 --focus-row 844 --focus-col 0 \
  | tee /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/boundary_A5_20261118.log
```

`--out` 必须是新目录；重跑改一个新名字。`--focus-row/col` 仅生成额外的 20 列数字窗口，**不缩小全量检查范围**。读取模型时只加载这一条 norm 权重，不加载完整模型。

正常完成显示 `ALIGN_OK` 和 `COMPUTED`。`COMPUTED` 仅说明参考计算完成，不能解释为精度通过。

回传：

```bash
cat /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/boundary_A5_20261118/summary.txt
```

`checks.jsonl` 保存每项完整统计；`focus.json` 保存第 844 行指定窗口及 8 个 rank 的本地贡献。如果失败，回传异常和已生成的 `summary.txt` / `error.json`。

## 3. A3 大包到位后，可在绿区复核 A3 基准

Git 已附带约 153 KiB 的 `green_route_20261117/reference/A3_l0_post_attention_norm.json`，由 A3 原 checkpoint 在 CPU 上导出，包含权重及 epsilon、原始 dtype、BF16 权重 SHA256。**只用于 A3 dump，不要用它替代 A5 checkpoint 的 norm 参数。**

若大包尚未解压，先按原文档运行 `green_route_20261117/prepare_data.sh`。然后：

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
python3 scripts/longcat2_950dt/check_attn0_moe_boundary.py \
  --run /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/A3_route_full_20261117_2110/A3_fixed \
  --params scripts/longcat2_950dt/green_route_20261117/reference/A3_l0_post_attention_norm.json \
  --out /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/boundary_A3_20261118 \
  --threads 4
```

蓝区已使用真实 A3 dump 跑过 CPU 对照，结果见 `green_route_20261117/reference/A3_boundary_cpu_baseline.json`：

- 复制张量在 8 个 rank 上相等，`moe.input` 与 `ffn0.input` 相等。
- `norm_from_stored_residual` 最大差 `0.001953125`。
- `fp64_once` 的 `reduce_plus_residual`、`fused_add_norm` 最大差均为 `0.00390625`。

**这些 A3 差值不是通用容差阈值。** 它们说明“高精度求和参考”“已舍入残差的 norm”本来就可能与真实 BF16 集合通信和融合算子不同，不能将非零差值直接判成 A5 故障。仍需结合完整分桶、差异位置、原值及不同候选模型的表现判断。

## 4. 与完整 A3 / A5 跨平台比较配合

边界 CPU 检查回答的是各自操作数下的计算差异；完整跨平台比较回答 A3/A5 的差异分布。A3 大包到位后继续运行原命令：

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
bash scripts/longcat2_950dt/green_route_20261117/compare_fixed.sh
```

若该比较的输出目录已存在，设置新目录后再执行，例如 `export LC_COMPARE_OUT=/mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/compare_fixed_20261118`。

优先分析 `reduce_plus_residual` 是否异常，再看融合 norm 参考。若候选模型无法解释，不会据此直接宣称 HCCL 或 RMSNorm 错误；缺失的真实归约中间值和运行时 norm 参数仍限制归因。本脚本不会触发重新采集。

## CPU 自检

```bash
python3 -m pytest -q test/manual/ascend/test_longcat_attn0_moe_boundary.py
```

覆盖 BF16 舍入中点、不同归约顺序的抵消、融合与已舍入残差归一化的区别、非有限值及最大差坐标、参数校验、混合 dump 和缺失 rank 拒绝；已在蓝区 CPU 环境通过 9 个用例。
