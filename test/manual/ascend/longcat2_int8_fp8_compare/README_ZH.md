# 950DT：修复版 DeepEP 下的 INT8 / FP8 配对评测

使用绿区当前已验证的 SGLang 和已安装的 `longcat2.0-deepep` 编译产物。
本工具与完整模型采集源码同在 `hanwlax/sglang` 的
`codex/longcat2-a5-full-fp8-capture` 分支。本次更新只增加评测工具和说明。
不安装 DeepEP、不启用激活探针。适用于四机 TP32 / attention DP2 /
EP32、DeepEP auto、AI_CPU、BF16 dispatcher、开 decode 图、开 NEXTN。

## 1. 从远端拉取代码

在绿区原容器中执行。首次获取时使用一个新目录：

```bash
export CAPTURE_SOURCE=/mnt/share/hanwlax/workspace/sglang-a5-full-capture
git clone --single-branch --branch codex/longcat2-a5-full-fp8-capture \
  https://github.com/hanwlax/sglang.git "$CAPTURE_SOURCE"
export TOOLS="$CAPTURE_SOURCE/test/manual/ascend/longcat2_int8_fp8_compare"
git -C "$CAPTURE_SOURCE" log -1 --oneline
test -f "$TOOLS/pair.py" && test -f "$TOOLS/eval_pair.py"
```

如果已经有此前采集分支的 checkout（也可以是此前按 `ae2d3eaa48f3` 创建的 detached
worktree），将 `CAPTURE_SOURCE` 改为它的实际目录，**用下面一组命令代替 clone**：

```bash
export CAPTURE_SOURCE=/mnt/share/hanwlax/workspace/sglang-a5-eliza-full
git -C "$CAPTURE_SOURCE" status --short
git -C "$CAPTURE_SOURCE" merge-base --is-ancestor ae2d3eaa48f3 HEAD || {
  echo '该目录不是此前完整模型采集代码，请改用上面的新目录 clone 命令'; exit 1;
}
git -C "$CAPTURE_SOURCE" fetch https://github.com/hanwlax/sglang.git \
  codex/longcat2-a5-full-fp8-capture
git -C "$CAPTURE_SOURCE" merge --ff-only FETCH_HEAD
export TOOLS="$CAPTURE_SOURCE/test/manual/ascend/longcat2_int8_fp8_compare"
git -C "$CAPTURE_SOURCE" log -1 --oneline
test -f "$TOOLS/pair.py" && test -f "$TOOLS/eval_pair.py"
```

仅快进更新；本地有冲突或分叉时停止处理，不执行 reset/clean。
绿区沿用已有的外网代理访问 GitHub；这不需要蓝区直连绿区，也不需要同步 zip。

`CAPTURE_SOURCE` 是版本化框架/采集工具所在目录；实际推理继续沿用已经测到
INT8=0.92 的 `run.sh` 中设置的源码或私有 runtime。不要因拉取工具而改变已验证的
`SGLANG_SOURCE_ROOT`，也不要重新生成旧的 trace runtime 来跑本轮自然生成评测。
如果原服务就直接使用此采集分支的普通 `python/`，可继续使用该目录。
两种模型必须使用同一个实际推理目录，工具会记录并对照其源码哈希。

## 2. 共同部署策略

本工具只包装当前 `run.sh` 的启动入口；SDK 初始化、网卡、overlap、缓存、内存、
batch size、NEXTN 步数和 token 数沿用该脚本。同一轮四台机器、两种模型均使用同一份模板。

复制当前**修复后评测使用的** `run.sh` 为 `run.compare.sh`，作为公共模板。
启动命令必须为直接的 `python3 -m sglang.launch_server`（`python` 也支持），
其余包装形式会在 prepare 时拒绝，避免误改。

确认公共模板明确包含以下设置；若已有对应项，不重复添加：

```bash
export HCCL_OP_EXPANSION_MODE=AI_CPU
export MOE_ENABLE_TOPK_NEG_ONE=1
export DEEPEP_HYBRID_DEPLOYMENT=1

# 在 python 启动参数中：
--tp-size 32 \
--attn-dp-size 2 \
--ep-size 32 \
--moe-a2a-backend deepep \
--deepep-mode auto \
--deepep-dispatcher-output-dtype bf16 \
```

图和 NEXTN 参数保留现状，勿换成旧的关图/关 NEXTN capture 脚本。
模板中的模型目录和量化参数会被以下规则覆盖，其他 CLI 参数不变：

| 参数 | INT8 | FP8 |
|---|---|---|
| model/tokenizer/draft 路径 | 指定的完整 INT8 目录 | 指定的完整 FP8 目录 |
| quantization | compressed-tensors | fp8 |
| speculative-draft-model-quantization | 移除显式参数，沿用服务默认推导 | fp8 |

需要 38 层完整模型。INT8 目录由用户填写，不能使用裁层目录。
源模板若有按 `MODEL_PATH` 分支选择其他部署参数的逻辑，需要去掉这些分支；
否则后面的共同参数检查会拒绝对照。

## 3. 冻结一次数据集和评测参数

以下变量在各机器容器中设置。`PAIR` 使用新目录，两种模型共用；
`TOOLS` 和 `PAIR` 应在四台机器上实际共享可见。若 `/mnt/share` 不是共享存储，
须将准备好的同一目录复制到四台机器同一路径，并在评测前汇集四份 `launch/rank*.json`。

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0
export TOOLS="$CAPTURE_SOURCE/test/manual/ascend/longcat2_int8_fp8_compare"
export PAIR="$PWD/int8_fp8_pair_01"
export SGLANG_SOURCE_ROOT=/mnt/share/hanwlax/workspace/sglang
# 如当前 run.sh 使用私有 runtime，这里也必须指向它，不能改回旧源码。
export PYTHONPATH="$SGLANG_SOURCE_ROOT/python"
export PYTHONUNBUFFERED=1
cp run.sh run.compare.sh
```

在 `run.compare.sh` 中核对上一节的公共参数，然后指定此前评测使用的完整 GSM8K
`test.jsonl`。不从 HTML 复制问题，避免字符编码变化。不使用只包含 200 道评测题的裁剪文件。

```bash
export GSM8K_DATA=/实际路径/gsm8k_test.jsonl
python3 "$TOOLS/pair.py" prepare \
  --run-sh "$PWD/run.compare.sh" \
  --data "$GSM8K_DATA" --out "$PAIR" \
  --count 200 --shots 5 --threads 4
```

只执行一次 prepare。若没有找到数据文件，可用当前 SGLang 的缓存下载工具取得路径：

```bash
python3 - <<'PY'
from sglang.utils import download_and_cache_file
print(download_and_cache_file(
    'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'))
PY
```

这沿用 legacy `run_eval`：数据集前 5 行作 few-shot，随后 200 行评测，Completion API、
temperature=0、top_p=1、max_tokens=2048、repeat=1、客户端并发 4。
如果要匹配之前的其他并发数，仅在 prepare 时修改 `--threads`，两种模型共用该值。
启动脚本和数据文件都会复制并哈希；后续不会重新下载或重新选题。

## 4. 启动 INT8：四台机器均执行

先用原有方式停止自己上一轮服务，确认这四台机器没有同一任务遗留 worker；
脚本不自动杀进程、不停止容器。无需改变修复后的 DeepEP 安装。

```bash
export INT8_MODEL=/实际完整INT8模型目录
python3 -u "$TOOLS/pair.py" start \
  --pair "$PAIR" --variant int8 --model "$INT8_MODEL" --run int8_r1
```

沿用 `run.sh` 自动识别本机 rank 的逻辑；所有机器使用同一 `int8_r1`。
控制台 `PAIR_LAUNCH` 为实际启动命令。
四台依次生成 `runs/int8_r1/launch/rank0.json` 至 `rank3.json`，
保存实际 argv、关键环境变量、模型配置、tokenizer 文件哈希、SGLang Python 源码哈希、
已安装 DeepEP Python 文件及扩展 `.so` 哈希。哈希用于一致性核验，不能单凭它证明补丁内容。
启动日志遵循公共模板的 tee 行为；若模板支持 `LOG_DIR`，自动指向本轮 `launch/serve_logs/`。

## 5. INT8 评测：仅一个客户端执行

在已初始化 CANN/ATB 环境的容器终端中，确认 `PYTHONPATH` 与服务实际源码一致。
等待服务就绪后运行，不要同时向该服务发送其他评测/采集请求。

```bash
set -o pipefail
export BASE_URL=http://141.61.45.105:10086
python3 -u "$TOOLS/eval_pair.py" \
  --pair "$PAIR" --run int8_r1 --eval-id gsm8k_200_r1 \
  --url "$BASE_URL" \
  2>&1 | tee "$PAIR/int8_r1_eval.log"
```

评测前读取实际 `/server_info` 核验模型、量化、并行、DeepEP、图和 NEXTN 配置；
保留当前安装版本的选题、提示词、评分、请求参数和重试行为。
不调用清缓存接口，不注入 cache salt，不修改采样 seed。
原版 `run_eval` 的 HTML 和总分 JSON 改存本轮目录，额外观察 Completion 请求和响应。

## 6. 换成 FP8，保持共同部署策略

INT8 评测结束后，停止自己这一轮四机服务，然后四台均执行：

```bash
python3 -u "$TOOLS/pair.py" start \
  --pair "$PAIR" --variant fp8 \
  --model /mnt/share/weight/LongCat-2.0-FP8 --run fp8_r1
```

服务就绪后，仅一个客户端执行：

```bash
set -o pipefail
python3 -u "$TOOLS/eval_pair.py" \
  --pair "$PAIR" --run fp8_r1 --eval-id gsm8k_200_r1 \
  --url "$BASE_URL" \
  --reference "$PAIR/runs/int8_r1/gsm8k_200_r1" \
  2>&1 | tee "$PAIR/fp8_r1_eval.log"
```

`--reference` 在发出正式请求前检查共同部署参数、关键环境、源码、DeepEP、
tokenizer 文件和评测脚本哈希。存在差异就停止；不要删除校验来得到分数。
tokenizer 文件可能存在仅元数据不同的情况；此时需要核对真实 tokenization 后再决定，
本工具不擅自将它认作相同。当前严格比较会拒绝该情况。

两种服务均用新启动后的第一次评测作为首轮对照。重复实验时对两种模型对称执行：
若在同一个服务上重复，改 `--eval-id gsm8k_200_r2`；若重启，改 `--run int8_r2/fp8_r2`。
不要比较一边冷启动首轮与另一边大量预热后的轮次而不记录区别。

## 7. 逐题对照与结果文件

```bash
python3 "$TOOLS/pair.py" compare \
  --left "$PAIR/runs/int8_r1/gsm8k_200_r1" \
  --right "$PAIR/runs/fp8_r1/gsm8k_200_r1" \
  --out "$PAIR/comparison_r1"
```

每轮评测目录保存：

| 文件 | 内容 |
|---|---|
| `questions.jsonl` | 每题 dataset 行号、prompt SHA256、完整 prompt/回答、gold、parsed、correct、停止原因、用量、错误和耗时 |
| `attempts.jsonl` | sampler 每次调用 Completion SDK 的参数、原始响应或异常；SDK 内部的 HTTP 重试不单独展开 |
| `summary.json` | 总分、完成数量、请求错误、空答、文本连续零、length 截断数量 |
| `prompts.json` | 按题目顺序固定的 200 个完整提示词及 SHA256 |
| `server_info.json` / `manifest.json` | 实际服务配置及四节点启动/二进制核验记录 |
| `gsm8k_LongCat-2.0.html/.json` | 原始 run_eval 报告和指标 |
| `eval_argv.json` / `run_eval_recorded.py` | 实际评测参数和仅重定位输出路径的 evaluator 副本 |

并发下 `questions.jsonl` 按完成顺序追加，**不能按文件行号配对**。
对照工具按完整 prompt 的 SHA256 配对，并输出：

- `comparison_r1/summary.json`：双方均对、仅 INT8 对、仅 FP8 对、双方均错的计数。
- `comparison_r1/paired.jsonl`：上述分类以及两边完整逐题记录。

有请求错误或缺题会保留数据、返回非零退出码；对照工具拒绝将其当作干净精度基线。
Completion API 未提供原始 output IDs 时 `token_zero_run=null`，不能用文本中的字符 `0`
代替 token ID 0。`text_zero_loop` 是连续至少 64 个字符 0 的检测，不是通用循环判定。
最终数值相等也不保证推理过程正确，原始回答已保存以便进一步分析。

在两组运行的 rank0/rank2 日志中分别核验实际 decode 图命中和 NEXTN accept len：

```bash
grep -h 'Decode batch' "$PAIR"/runs/int8_r1/launch/serve_logs/*.log | tail -n 8
grep -h 'Decode batch' "$PAIR"/runs/fp8_r1/launch/serve_logs/*.log | tail -n 8
cat "$PAIR/comparison_r1/summary.json"
```

如果公共模板将日志写到固定目录，请使用模板的真实日志路径；`server_info` 只证明配置开启，
不能代替逐 batch 的图命中证据。

## 本地验证范围

已做 CPU 工具验证：实际 legacy `run_eval`、`GSM8KEval`、`CompletionSampler` 源码，
配合模拟 Completion 响应和依赖，验证并发逐题保存、原始评分一致、错误请求保留、
INT8/FP8 CLI 差异和配置漂移拒绝。另有格式、语法和包装 Bash 语法检查。
没有在绿区 NPU 起服或运行 GSM8K；不宣称新一轮分数。KG 本次连接不可用，
部署兼容性以本地源码检查和绿区运行回执为准。[未核实：绿区实际执行]
