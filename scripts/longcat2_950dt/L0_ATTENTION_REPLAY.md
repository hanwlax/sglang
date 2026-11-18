# L0 第一段 Attention 离线重放

两层固定路由的 Control 冷 prefill 有 852 行、TP8。使用已有 dump 和 checkpoint，所有矩阵乘法在 CPU 上运行。216 的 A3 配置显式关闭了 self_attn 量化，Q/KV 输入投影及 O 投影的 checkpoint 权重为 BF16。950DT 使用原生 E4M3、128×128 权重块和每行 128 列一组的激活量化。

## 绿区

把 eTrans 中的 `a3_l0_qkv.pt.gz` 放进绿区容器可读目录，将路径传给脚本。

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
git fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git merge --ff-only FETCH_HEAD
bash scripts/longcat2_950dt/green_route_20261117/compare_l0_attention.sh \
  /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/a3_l0_qkv.pt.gz
```

脚本校验来自 216 的压缩文件、固定路由凭据、两边所有 TP 的完整输入。Q/KV 输入投影权重在 TP 间复制，同一份输入只需要计算一次。两块投影分别是 [852,1536] 和 [852,576]，观测到的 O 输出则逐个 TP 检查全部 [852,8192] 元素。每个比较打印全张量绝对误差分桶、首个和最大差异坐标，以及第 844 行、第 0–19 列两侧的值和有符号差。Q、KV 和 O 的列属于不同的输出空间。

| 输出 | 含义 |
|---|---|
| `checkpoint_fp64` | 同一份 BF16 输入分别乘 A3 BF16 权重和反量化的 950DT FP8 权重，均用 FP64 累加 |
| `with_fp8_activation_quant` | 恢复 950DT 的 CPU 激活量化模型，再与同一 A3 参考比较 |
| `fp8_activation_effect_only` | 同一 950DT 权重，量化激活前后两次 CPU 计算的逐元素差 |
| `output_bf16` | 两个 CPU 投影各自舍入至 BF16 |
| `QKV_DUMP` | 发现旧 rank0 融合投影 dump 时检查完整输入。输入匹配后再检查权重字节和 scale，比较当次实测投影与 A3 CPU、当次实测投影与其实际操作数的 CPU 重放 |
| `O_INVENTORY` | 列出各边有输入的 TP 和可交叉重放的交集 |
| `attn0.local_output / observed` | 两侧已保存的真实本地输出，TP0–7 分别对照 |

入口会检查 `/tmp/longcat_attention_tail_rank0_*.pt`。历史 tail 仅在 L0、V 输出与 O 输入相符且整个 TP0 O 输出与固定采集逐元素相同时参与比较。若有多份可选 `/tmp/longcat_qkv_fp8_rank0_*.pt`，用 `QKV_DUMP=/实际文件` 指定一份。现有 A3 固定采集缺少 O 投影输入，两侧输入配对为空时，输出只给出实测边界差异。仅凭这个边界和 Q/KV 的差异无法分摊 O 输入、O 权重各自的贡献。

## 蓝区重建参考

216 本次文件在 `/home/hanwlax/workspace/progress/longcat2.0/route_runs/l0_projection.3zKnXh/a3_l0_qkv.pt.gz`，SHA-256 为 `77af206bf3997823394679ff8d4f2169b6a700670890665f7df2bc48125473e8`。包内两块投影均保留 FP64。重新生成时从仓库根目录运行，输出路径须是新文件。

```bash
python3 scripts/longcat2_950dt/replay_l0_attention.py a3 \
  --model /home/l00951279/longcat2/LongCat-2.0-INT8-part \
  --a3 /home/hanwlax/workspace/progress/longcat2.0/route_runs/A3_route_full_20261117_2110/A3_fixed \
  --reference ./a3_l0_qkv.pt.gz
```

## 38 层 Eliza 全模型准备

此实验另用完整模型、Eliza 的两个固定前缀、真实路由，包含 prefill 和 decode。950DT 四机启动、重放和完整性校验沿用[全模型采集说明](https://github.com/hanwlax/sglang/blob/ae2d3eaa48f3f2f86420b1ad8bfa3fb0d51a3bc7/test/manual/ascend/longcat2_baseline/A5_ELIZA_FULL_CAPTURE_ZH.md)。

217 的首批 A3 参考在 `/home/hanwlax/workspace/progress/longcat2.0/a3_window_20261117_5cf81e6/archives/eliza_cpu_pack.ghO9Uc/a3_eliza_first.zip`。包内包含两个前缀的两次正式重放、DP0 全部 16 个 attention TP 的每个 decode 边界，以及 prefill 的每层 `layer.output`、`layer.residual` 和 logits。每条逻辑记录保留行号、位置、语义及分片信息，张量字节按原始 dtype、形状和 SHA-256 去重。首包 ZIP 和成员 CRC 已校验，首包 SHA-256 为 `c12b6e5265955144ac2ed8ca9d99e4387d274ff9cd086d5acd7e0a03989c89d8`。

其余 prefill 边界可按层补包。下面在蓝区归档所在文件系统执行，`--extend-remainder-layer` 取 `-1`（全局）或 0–37，输出放新路径。

```bash
root=/home/hanwlax/workspace/progress/longcat2.0/a3_window_20261117_5cf81e6/archives
python3 scripts/longcat2_950dt/pack_eliza_boundaries.py \
  --dump "$root/a3_capture_dp0_20261117_2044/dumps" \
  --extend-remainder-layer 0 --out ./a3_eliza_extend_L0.zip
```

全模型先核验 950DT 自身两次重复，跨平台按同一前缀、位置、全局 head/专家、有效行和分片语义对齐。TP64/EP64 与 TP32/EP32 产生的不同局部分片需要重组后再对照。首包和逐层补包都覆盖边界文件，block 0/7/37 内部探针另在原归档中。两层 Control 的数值结果不能替代这轮比较。
