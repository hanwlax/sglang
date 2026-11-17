# LongCat 激活矩阵查看器

两台电脑各自读取本地 `.pt`，使用相同的层、采集边界、TP、行列坐标及颜色范围做肉眼对照。程序只在 CPU 上读取文件，不修改采集数据，也不调用模型或 NPU。网页资源全部随源码提供，不使用 CDN 或外部网页服务。

## 1. 获取代码与依赖

代码位于 `hanwlax/sglang` 的 `codex/longcat2-route-lock-full` 分支。在已有的独立 checkout 中，确认没有未处理的本地修改后更新：

```bash
git status --short
git fetch https://github.com/hanwlax/sglang.git codex/longcat2-route-lock-full
git merge --ff-only FETCH_HEAD
```

运行需要 Python 和支持 `torch.load(weights_only=True)` 的 PyTorch。原 SGLang 容器已有 PyTorch，无需额外安装。普通电脑可使用单独的 Python 环境安装 CPU 版本：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

无需安装 Streamlit、Matplotlib、Node.js 或浏览器测试工具。

## 2. 两边启动

在各自源码根目录执行。默认监听 `127.0.0.1:8765`，浏览器访问 `http://127.0.0.1:8765`。两台电脑可以使用相同端口；若在同一台电脑启动两个实例，第二个指定 `--port 8766`。

A3 本次固定路由数据：

```bash
python3 scripts/longcat2_950dt/view_matrix.py \
  --root /home/hanwlax/workspace/progress/longcat2.0/route_runs/A3_route_full_20261117_2110/A3_fixed \
  --label A3
```

950DT 本次固定路由数据：

```bash
python3 scripts/longcat2_950dt/view_matrix.py \
  --root /mnt/share/hanwlax/workspace/progress/longcat2_0/route_runs/A5_fixed_20261117 \
  --label 950DT
```

`--root` 接受单轮采集目录，或其中的 `full/`。不要指向同时包含多个实验的总目录。程序会按文件名建立层、边界和 TP 的下拉菜单，一次仅缓存一个矩阵。

如果 `.pt` 已经在电脑本地，也可以只打开一个文件（Windows 上使用 `python`，并替换成本机路径）：

```bash
python3 scripts/longcat2_950dt/view_matrix.py \
  --file /实际路径/full_L0_layer.output_TP0.pt --label A3
```

支持普通二维 Tensor，以及本次包含 `tensor`、`rows` 等元数据的采集字典。非二维文件会明确报错，不自动 reshape 或猜测维度语义。只打开可信的本地采集文件，加载始终使用 `weights_only=True`。

## 3. 浏览器在电脑、数据在服务器时

可以在服务器运行查看器，在同一区域的电脑建立 SSH 转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 用户名@同区服务器IP
```

然后在电脑打开 `http://127.0.0.1:8765`。这要求 SSH 端的 `127.0.0.1:8765` 能访问查看器；如果查看器位于独立网络的容器，需按现有容器网络方式暴露端口或转发到容器地址。

也可在本区可信网络上给启动命令追加 `--host 0.0.0.0`，浏览器访问 `http://服务器IP:8765`，并确认容器端口映射。此查看器没有登录认证，默认仅监听本机。蓝区、绿区分别启动和访问，不建立跨区连接。

## 4. 操作步骤

1. 两边都选择层 `0`、边界 `layer.output`、TP `0`，即第一层这一采集边界的对应矩阵。文件不同 TP 不能混用；`layer.residual` 是另一个边界。
2. 热力图点击“整矩阵热力图”观察总体分布，悬停可看色块覆盖的原始行列范围。点击色块进入局部窗口。
3. 输入起始矩阵行、起始列、行数、列数，点击“显示窗口”。坐标从 0 开始；窗口尾部超出矩阵时只显示实际剩余范围，并更新框内行列数。起始坐标越界会报错。
4. 切换“数字表格”，每个单元格显示原始值；可选科学计数或小数格式、0–16 位小数。悬停显示更完整数值，点击单元格固定坐标信息。表格最多 64×64；从整图切换时可先点击“局部 16×12”。
5. A3 页面点击“复制当前参数”，在另一台电脑下方文本框粘贴并“应用参数”。参数仅包含选择、坐标、颜色及数字格式，不含张量数据或路径，也不会连接两台电脑。参数也保存在本页 URL 的 `#` 后，可留作书签。

如果无法自动访问剪贴板，界面会选中文本，请按 Ctrl+C。跨区时按既有文本传递流程复制参数。不存在的层/TP 或越界窗口会报错，不会静默换到其他矩阵。

数字表格显示“矩阵行 / token行”：通常两者相同，但 `lm_head.logits` 的 `[1,163840]` 文件中，矩阵行 0 对应 token 行 851。`--file` 打开的裸 Tensor 没有行映射，按本地矩阵行编号显示。

## 5. 如何解读热力图

- 默认两边固定颜色范围 `[-1,1]`。负端蓝色，中点近白，正端红色；范围不对称时，中点颜色对应上下限的算术平均。
- 超出颜色范围的值使用端点颜色；数值没有被截断。窗口统计和数字表格仍显示原值。
- “按当前窗口调整颜色”只修改本页颜色范围；调整后必须把新参数同步到另一台电脑，避免各自自动缩放造成误判。
- 热力图最多生成 256×512 个色块。大窗口中每块取**绝对值最大的有限元素，并保留符号**；界面明确显示一个块覆盖多少行、多少列。它不是均值，也不代表块中所有元素相同。
- 局部窗口不超过 256 行、512 列时，每个色块对应一个元素。NaN/Inf 单独用紫色标记；即使块中同时有正常数值，也保留异常标记。
- 肉眼查看适合观察分布和局部数值，不能替代 `compare_full_route_tensors.py` 的全部元素扫描。可先从其 `MAX` 中取层、边界、TP 和 `coordinate`，再把附近窗口填到两边查看器中。

## 6. 验证与退出

CPU 自检（包含临时本机 HTTP 端口测试，不需要 NPU）：

```bash
python3 scripts/longcat2_950dt/check_matrix_viewer_cpu.py
```

开发验证覆盖生产形状、边缘窗口、汇总块与填充、NaN/Inf、整数精度、logits 行映射和 HTTP 参数检查。浏览器交互另在真实 A3 `[852,8192]` 文件及受控合成差异上验证；这些检查不构成 950DT 精度结论。

使用结束后，在运行查看器的终端 Ctrl+C 即可。所有读取均在 CPU 完成，不会改写 `.pt`。
