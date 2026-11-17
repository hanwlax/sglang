# A3 / A5 完整有效边界元素采集步骤

本指南先采完整38层模型的block0、1，不裁层；两边attention TP均为16。
新模式schema5，旧schema4比较脚本不适用。跨平台比较脚本待真实数据落盘后提供。
A3仍须先经自然生成GSM8K验证，不能以强制前缀采集证明准确率。

## 1. 每台容器获取同一个源码版本

在各自原来的Ascend容器内执行。不要checkout/reset正在使用的共享源码。
以下TASK必须是新目录；四机不保证共享，需要各机执行并核对相同commit。

A3的209、212、216、217分别设置：

```bash
REPO=/home/hanwlax/workspace/sglang
TASK=/home/hanwlax/workspace/progress/longcat2.0/full_elements_20261117_v1
PROFILE=a3-dp4
```

A5的.105、.113、.121、.122分别设置：

```bash
REPO=/mnt/share/hanwlax/workspace/sglang
TASK=/mnt/share/hanwlax/workspace/progress/longcat2_0/full_elements_20261117_v1
PROFILE=a5-dp2
```

然后两边都执行：

```bash
mkdir "$TASK"
git -C "$REPO" fetch https://github.com/hanwlax/sglang.git codex/longcat2-a3-dp4-baseline
git -C "$REPO" worktree add --detach "$TASK/source" FETCH_HEAD
git -C "$TASK/source" rev-parse HEAD
mkdir "$TASK/tools"
cp -a "$TASK/source/test/manual/ascend/longcat2_baseline/." "$TASK/tools/"
cd "$TASK/tools"
```

核对HEAD是包含full-boundary改动的新提交，不能停留在813ffa48f884。
生成器仍固定推理基点c0e9d9a0cc6b；它不自动合并绿区共享源码上的未提交数学实现改动。
如绿区Git不能拉取，按既定蓝→同步工具→绿区流程传入同一提交的工具及Git基点，不直接蓝绿互传。

## 2. 生成新runtime与CPU检查

```bash
# 上一步PROFILE=a3-dp4或a5-dp2；runtime路径使用绝对路径。
if [ "$PROFILE" = a3-dp4 ]; then
    RUNTIME="$TASK/tools/runtime_full_a3"
else
    RUNTIME="$TASK/tools/runtime_full_a5"
fi
python3 prepare_runtime_dp4.py --source "$TASK/source" --mode trace \
    --trace-topology "$PROFILE" --full-boundary-layers 0 1 --out "$RUNTIME"
python3 verify_full_boundary.py > "$TASK/tools/full_capture_cpu_check.log" 2>&1
tail -n 1 "$TASK/tools/full_capture_cpu_check.log"
```

应看到FULL_BOUNDARY_CPU_PASS。receipt.json记录所有相关文件SHA。
每台都是本地构建时，receipt的路径字段可以不同，但同profile的源码hash应相同。
改helper/生成器后要换新runtime路径重建、重启；改tools目录外层helper不改变已加载代码。

## 3. A3先评测，再启动采集服务

若本套完整A3 baseline还没有GSM8K结果，四机先额外生成eval runtime：

```bash
python3 prepare_runtime_dp4.py --source "$TASK/source" --mode eval \
    --out "$TASK/tools/runtime_dp4_eval"
# 四机NODE_RANK：209=0，212=1，216=2，217=3。
NODE_RANK=0
bash run_a3_full_baseline.sh "$NODE_RANK" a3_eval_full_v1 eval --dry-run
# 所有64张逻辑卡已交接且空闲后启动：
bash run_a3_full_baseline.sh "$NODE_RANK" a3_eval_full_v1 eval
```

服务就绪后，216另一个容器终端进入同一TASK/tools，执行：

```bash
bash run_gsm8k.sh a3_gsm8k_200_full_v1
```

保留evaluations/a3_gsm8k_200_full_v1，检查score及逐题异常。确认基线质量后，
通过本任务所属终端停止eval服务，再四机分别启动新trace RUN，不停止其他人的服务：

```bash
cd /home/hanwlax/workspace/progress/longcat2.0/full_elements_20261117_v1/tools
NODE_RANK=0 # 209=0，212=1，216=2，217=3，按本机修改
RUNTIME="$PWD/runtime_full_a3" bash run_a3_full_baseline.sh "$NODE_RANK" a3_capture_01 trace --dry-run
RUNTIME="$PWD/runtime_full_a3" bash run_a3_full_baseline.sh "$NODE_RANK" a3_capture_01 trace
```

## 4. A5四机启动采集服务

在绿区原容器、设备已交接释放之后，四机分别执行（无需改原run.sh）：

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0/full_elements_20261117_v1/tools
NODE_RANK=0 # .105=0，.113=1，.121=2，.122=3，按本机修改
RUNTIME="$PWD/runtime_full_a5" bash run_a5_full_boundary.sh "$NODE_RANK" a5_capture_01 trace --dry-run
RUNTIME="$PWD/runtime_full_a5" bash run_a5_full_boundary.sh "$NODE_RANK" a5_capture_01 trace
```

默认模型/mnt/share/weight/LongCat-2.0-FP8；TP32/DP2/EP32；DeepEPauto、AI_CPU；
关prefill/decode图、关NEXTN、关radix；MOE_ENABLE_TOPK_NEG_ONE=1、DEEPEP_HYBRID_DEPLOYMENT=1。
HCCL默认enp34s0f1、Gloo默认data0.3001，可用同名环境变量覆盖。
启动脚本只通过CPU合成路径dry-run，尚无A5 NPU启动成功声明；遇报错保留四机首个完整错误。

## 5. 两边重放同一输入

各自服务就绪后，只有一个客户端串行发请求，期间不运行其他评测或请求。
先DP0、length1、repeat2；包含r0预热、r1/r2测量。预热不落盘完整张量。

A3在216另一个终端：

```bash
cd /home/hanwlax/workspace/progress/longcat2.0/full_elements_20261117_v1/tools
python3 replay_fixed_decode_dp4.py --topology a3-dp4 \
    --control fixtures/control_852.json --url http://192.168.25.209:10096 \
    --dps 0 --lengths 1 --repeat 2 --timeout 3000 \
    --out "$PWD/results/a3_capture_01/client"
```

A5在.105另一个终端：

```bash
cd /mnt/share/hanwlax/workspace/progress/longcat2_0/full_elements_20261117_v1/tools
python3 replay_fixed_decode_dp4.py --topology a5-dp2 \
    --control fixtures/control_852.json --url http://127.0.0.1:10086 \
    --dps 0 --lengths 1 --repeat 2 --timeout 3000 \
    --out "$PWD/results/a5_capture_01/client"
```

852个原始token，强制prefill输出70115；随后真实decode读取70115、position852。
r1/r2对应关系由manifest的dp/length/rep匹配，不要求两边随机RID相同。
超时会停止客户端，不叠加重发，先处理未完成请求。
如需要position859，用新服务RUN/客户端输出目录和--lengths 8。

## 6. 日志和需要收集的文件

A3每台tools目录：

```bash
grep -hE 'BASELINE_RUNTIME|LONGCAT_FIXED_V4_WORKER|LONGCAT_FIXED_V4_FORCE|LONGCAT_FULL_BOUNDARY|Traceback|RuntimeError' results/a3_capture_01/dumps/serve_rank*.log
du -sh results/a3_capture_01/dumps
```

A5替换a3_capture_01为a5_capture_01。数据布局：

```text
tools/
  runtime_full_a3/receipt.json             # A5为runtime_full_a5
  results/a3_capture_01/                   # A5为a5_capture_01
    client/manifest.json                  # 客户端主机
    client/*.request.json
    client/*.response.json
    launch_rankN.json                     # 每台一份
    dumps/worker.tpN.json                  # A3共64份，A5共32份
    dumps/serve_rankN.log                  # 四机启动/探针日志
    dumps/lcfix4-....tpN.EXTEND.pt          # A5前缀lcfix2；完整索引
    dumps/lcfix4-....tpN.EXTEND.json        # 可直接阅读的相同索引
    dumps/lcfix4-....tpN.EXTEND.full/       # 原始元素文件
      layer-1.embedding.output.pt
      layer0.attn0.input.pt
      layer0.attn0.local_output.pt
      layer0.router.expert_ids.pt
      layer0.moe.branch_output.pt
      layer0.ffn0.local_output.pt
      layer0.layer.output.pt
      layer0.layer.residual.pt
      ...
      layer1.*.pt
      layer2.attn0.input.pt                # layer1的消费边界
      layer2.attn0.residual.pt
      layer-1.final_norm.output.pt
      layer-1.lm_head.logits.pt
    dumps/lcfix4-....tpN.DECODE.pt/.json
    dumps/lcfix4-....tpN.DECODE.full/...
```

有有效行的stage文件中`values`就是原dtype的完整选定张量，不是统计摘要。
无有效行的rank/stage记录no_live_rows，没有values文件。不要据此补零。
没有完成索引的.full目录视为中断证据，不能算完成数据。

只跑DP0、length1、repeat2时，每个平台应有64个完成forward索引（16rank×2repeat×2mode），
不是64个stage文件。A3 DP0的激活在209的TP0–15；A5 DP0横跨.105/.113的TP0–15。
即使其他主机没有激活文件，也须保存其worker收据、launch和日志。

各平台内部将同一RUN四机文件汇总到新目录，保留文件名；同名碰撞先核对SHA，不覆盖。
另外保留模型config.json和safetensors index的SHA/副本及runtime receipt。
A3可给蓝区路径直接分析；A5文件留绿区，后续把A3数据传入绿区，用脚本就地比较，回传文字摘要。

## 7. 全层与内部详细输出的能力边界

1. 全38层边界的完整有效元素：另建runtime，在两边生成命令中把
   `--full-boundary-layers 0 1`改成`--full-boundary-layers $(seq 0 37)`。
   改环境变量不会改变已经生成的层列表。每侧另起新RUN，其他协议相同。
2. A3 Attention内部详细输出：另建旧详细模式runtime：

```bash
python3 prepare_runtime_dp4.py --source "$TASK/source" --mode trace \
    --out "$TASK/tools/runtime_detail_a3"
LC_INTERNAL_LAYERS=all RUNTIME="$TASK/tools/runtime_detail_a3" \
    bash run_a3_full_baseline.sh "$NODE_RANK" a3_detail_all_01 trace
```

   四机同样配置；重放用--topology a3-dp4，输出新client目录。
   该模式全38层边界仍只抽首/中/末行，Attention内部在r1、默认全局rank0/16/32/48采集，
   即每个被选中的Attention TP rank只覆盖自己的4个头。保存a_proj、norm_qb、qk_bmm、
   rope、sparse、tail；文件在dumps/internal/，含物理cache需按有效索引过滤。
3. 全边界模式没有内部hook：设置LC_INTERNAL_LAYERS=all也不会额外抓内部。
   A5新DP2入口当前只支持全边界，尚未适配旧内部详细模式。
4. “内部详细”也不等于全模型所有算子：MoE专家GEMM、denseFFN内部GEMM以及FP8
   量化器的全部实际操作数并未全量采集。后续依据边界差异再扩展具体算子。
