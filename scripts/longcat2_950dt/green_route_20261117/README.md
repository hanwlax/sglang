# 950DT：使用 2026-11-17 A3 实测参考进行固定路由采集

代码和小文件都在 `hanwlax/sglang` 的 `codex/longcat2-route-lock-full` 分支：

- `reference/A3_reference.json`：约 493 KiB，实际 A3 两层、852 token 的专家 ID 和路由权重。
- `reference/control.json`：约 4.7 KiB，采集时使用的准确输入 token。
- `reference/A3_self_check.json`：A3 自对照结果。416 对完整张量、2,164,000,256 个元素全部相等，无非有限值；元素计数包含 rank 间重复值。
- `reference/archive.sha256`：大数据包的校验值。

完整张量不入 Git：`A3_route_full_20261117_2110.tar.gz` 通过 eTrans 传入。解压后约 8.3 GiB，包含 A3 自然组、固定组、路由记录、日志及逐元素自对照结果。**启动和重放只需 Git 文件；比较需要完整数据包。**

本轮仍是两层、TP8、DP1、attention DP1、EP1，关闭 DeepEP、图、NEXTN、radix cache，A5 使用 AI_CPU。模型默认 `/mnt/share/weight/LongCat-2.0-FP8-cut`。不要改成生产 TP32/EP32 或完整模型配置，否则无法与本次 A3 数据对齐。

## 1. 获取源码与小参考文件

在绿区原容器中执行。独立目录已存在时，先检查有无本地修改；不要强制覆盖。

```bash
git -C /mnt/share/hanwlax/workspace/sglang fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
```

若还没有独立 checkout：

```bash
git -C /mnt/share/hanwlax/workspace/sglang worktree add --detach /mnt/share/hanwlax/workspace/sglang-longcat-route-lock FETCH_HEAD
```

若已有独立 checkout，在确认 `git status --short` 没有本地修改后更新：

```bash
git -C /mnt/share/hanwlax/workspace/sglang-longcat-route-lock status --short
git -C /mnt/share/hanwlax/workspace/sglang-longcat-route-lock fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git -C /mnt/share/hanwlax/workspace/sglang-longcat-route-lock merge --ff-only FETCH_HEAD
```

本次交付只增加脚本和参考数据，实际运行的模型、探针及对比实现必须与 A3 采集版本 `4fd8d9febb95` 相同；脚本会自动检查这些文件并核验小参考文件 SHA256。

## 2. 启动固定路由服务（终端一）

正常停止原来的诊断服务，确保有 8 个空闲 NPU。默认可见设备 0–7，端口 10096。保留至少 20 GiB 可用磁盘用于 A3 解压和 A5 新采集。

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
bash scripts/longcat2_950dt/green_route_20261117/start_fixed.sh
```

输出目录固定为：

```text
/mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/A5_fixed_20261117
```

目录已存在会停止，不会覆盖。重跑时在**所有相关终端**设置同一个新目录，例如 `export LC_A5_RUN=/mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/A5_fixed_20261117_retry1`。若修改 `HTTP_PORT`，启动与重放终端必须一致。

## 3. 重放和核验（终端二）

服务就绪后，只执行一次：

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
bash scripts/longcat2_950dt/green_route_20261117/replay_fixed.sh
```

应出现 `REPLAY_OK`（416 个完整张量、16 条路由记录）和 `AUDIT_OK`（模式 `ids_weights`）。重放直接使用 Git 中的固定输入，不需要旧的 `Control.json` 或客户端 RID。

请求只生成一个 token，不进入 decode。专家 ID 和缩放后的路由权重来自 A3，替换发生在 identity expert 处理前。每个服务实例仅发送这一次匹配输入的请求。完成后在启动终端 Ctrl+C 正常停止诊断服务、释放 NPU。

## 4. 传入并校验完整 A3 数据

在本机 `D:\workspace\eTrans\longcat_route_20261117` 中只放大参考数据包和它的校验文件。通过同步工具“增量同步”到绿区 PC，再上传到绿区原容器能访问的目录。

以下假设包已上传为 `/mnt/share/hanwlax/workspace/progress/longcat2_0/A3_route_full_20261117_2110.tar.gz`；实际放在别处时仅修改参数。

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
bash scripts/longcat2_950dt/green_route_20261117/prepare_data.sh \
  /mnt/share/hanwlax/workspace/progress/longcat2_0/A3_route_full_20261117_2110.tar.gz
```

脚本先验证压缩包 SHA256，再解压并校验内部文件，最后确认大包与 Git 中的路由参考逐字节一致。预期 `A3_DATA_OK`。若之前已完整解压，直接在解压目录运行 `sha256sum --quiet -c SHA256SUMS`，不要覆盖旧目录。

## 5. 完整逐元素比较（CPU）

```bash
cd /mnt/share/hanwlax/workspace/sglang-longcat-route-lock
bash scripts/longcat2_950dt/green_route_20261117/compare_fixed.sh
```

会先核验两边实际采用的路由，再以 `--require-fixed` 比较 A3_fixed 与 A5_fixed：52 个边界 × 8 个 rank，所有保存元素均参与计算，不抽样、不计算 L2。

结果目录为 `route_runs/compare_fixed_20261117`，位于进度目录下。`summary.txt` 按边界给出最大绝对差、首个差异、最大差异坐标、各误差区间计数；`per_tensor.jsonl` 保存逐 rank 结果。整数 ID 只比较相等性，NaN/Inf 单独计数。重跑比较需设置新的 `LC_COMPARE_OUT`。

## 6. 回传文字摘要

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
cat route_runs/A5_fixed_20261117/replay.log
cat route_runs/A5_fixed_20261117/audit.log
cat route_runs/compare_fixed_20261117/summary.txt
```

不要上传绿区 `.pt` 文件。先看 embedding 和 Attention 输入是否已经不同，再沿两个层的 Attention、MoE、FFN、层输出、final norm 和 logits 定位误差变化。固定专家 ID 不会使 A3 INT8 和 A5 FP8 的权重、隐藏输入变成相同张量，不能仅凭跨平台误差直接断定 A5 算错。
