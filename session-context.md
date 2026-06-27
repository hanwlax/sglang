╭─── Claude Code v2.1.193 ─────────────────────────────────────────────────────╮
│                                                    │ Tips for getting        │
│              Welcome back Susan Musa!              │ started                 │
│                                                    │ Run /init to create a … │
│                       ▐▛███▜▌                      │ ─────────────────────── │
│                      ▝▜█████▛▘                     │ What's new              │
│                        ▘▘ ▝▝                       │ Added `autoMode.classi… │
│   Opus 4.6 · Claude Pro · Bambirqu@alumni.com's    │ Added auto-mode denial… │
│   Organization                                     │ Added `claude_code.ass… │
│           /home/hanwlax/workspace/sglang           │ /release-notes for more │
╰──────────────────────────────────────────────────────────────────────────────╯


  Searched for 1 pattern (ctrl+o to expand)
  ⎿  Read ../progress/pcg/sdpa_microbench.py (64 lines)
  ⎿  Read python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_b
     ackend.py (349 lines)

● Let me check the profiler-run server logs for startup warmup evidence to pin
  down the exact fwd counter offset.

  Thought for 5s, searched for 2 patterns (ctrl+o to expand)
  ⎿  Loaded .claude/rules/no-dataclasses.md

● Now let me look at how the fwd counter works in both runners to determine the
  exact offset. Let me check the Eager runner code.

  Read 2 files (ctrl+o to expand)

● Now let me check which code path controls the fwd counter — is it only in the
  else branch (i.e., only extend, not decode)?

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Good — the _eager_fwd_count increment is inside _execute_extend (line 253+),
  not _execute_decode. So only extend forwards count. Now let me check what the
  shape-log run tells us about the exact sequence of extend forwards for Eager.

  Searched for 1 pattern (ctrl+o to expand)

● Now let me count how many SDPA calls correspond to each extend forward. Each
  extend forward has 10 SDPA calls (10 full-attention layers). Let me count the
  groups.

  Thought for 55s, searched for 2 patterns (ctrl+o to expand)

● Let me verify the actual FA kernel durations from the profiler data to confirm
  our hypothesis.

  Searched for 4 patterns (ctrl+o to expand)

● Confirmed: Eager fwd#10 has FA at ~621-638us (10 calls), PCG fwd#10 has FA at
  ~2210-2245us (only 2 calls shown). The counts differ — Eager has 10 FA calls
  (one per full-attention layer), but PCG shows only 2. Let me check further.

  Searched for 2 patterns (ctrl+o to expand)

● Both have exactly 10 FA calls (matching 10 full-attention layers). Let me see
  all 10 FA durations for PCG.

  Searched for 2 patterns (ctrl+o to expand)

● Crystal clear. All 10 FA kernels in PCG are ~2230us (chunk2, seq_kv=8192), all
  10 in Eager are ~627us (chunk1, seq_kv=4096). The 3.55x ratio is exactly
  2230/627 = 3.56x, consistent with the O(n²) scaling between seq_kv=8192 vs
  4096 with causal masking.

  Now let me determine the exact fwd counter mapping. The Eager extend forwards
  are:
  - fwd#1,#2 = warmup (shape=80, seq_kv=80)
  - fwd#3 = serving chunk1 (shape=4096, seq_kv=4096)
  - fwd#4 = serving chunk2 (shape=4096, seq_kv=8192)
  - fwd#5 = chunk1, fwd#6 = chunk2, ...
  - fwd#10 = even → chunk2 → but profiler shows FA=627us (chunk1!)

  Wait, that contradicts. Let me reconsider. fwd#10 with warmup offset=2:
  serving fwd = 10-2 = 8th, which is even → chunk2. But Eager profiler shows
  chunk1 timing. Let me check if there are more warmup forwards or if the
  counter increments differently.

  Thought for 48s, searched for 1 pattern (ctrl+o to expand)

● The interleaved TP output makes exact grouping messy, but the conclusion is
  definitive. Let me also check the PCG shape log to confirm PCG doesn't have
  warmup SDPA calls.

  Thought for 45s, searched for 1 pattern (ctrl+o to expand)

● The PCG shape log reveals a crucial difference: PCG has an extensive graph
  capture phase with many SDPA calls at various seq_kv sizes (128 down to 128,
  capturing all 32 prefill batch sizes), THEN a warmup at seq_kv=80, THEN
  serving. Eager has no capture phase, just 2 warmup forwards then serving.

  This means _pcg_fwd_count and _eager_fwd_count at the same value (#10)
  correspond to completely different points in the serving timeline, landing on
  different chunk types.

  Root cause definitively confirmed: The "3.55x FA slowdown" is a measurement
  artifact:

  ┌───────────────────┬───────────────────────────────┬────────────────────┐
  │                   │          PCG fwd#10           │    Eager fwd#10    │
  ├───────────────────┼───────────────────────────────┼────────────────────┤
  │ Chunk type        │ chunk2 (seq_kv=8192)          │ chunk1             │
  │                   │                               │ (seq_kv=4096)      │
  ├───────────────────┼───────────────────────────────┼────────────────────┤
  │ FA kernel         │ ~2230us                       │ ~627us             │
  ├───────────────────┼───────────────────────────────┼────────────────────┤
  │ Ratio             │ 2230/627 = 3.56x              │ baseline           │
  ├───────────────────┼───────────────────────────────┼────────────────────┤
  │ Expected O(n²)    │ 8192²/4096² × causal_factor ≈ │ baseline           │
  │ ratio             │  3.6x                         │                    │
  └───────────────────┴───────────────────────────────┴────────────────────┘

  The micro-benchmark proved both NPUs have identical FA performance:
  - chunk1: PCG NPU ~712us, Eager NPU ~738us
  - chunk2: PCG NPU ~2291us, Eager NPU ~2350us

  Now let me update the memory with the corrected findings and revised overhead
  estimate.

  Recalled 2 memories, wrote 2 memories (ctrl+o to expand)

● 调查结论总结：

  FA 3.55x 慢的根因已定位为 profiler 测量伪影，不是真实性能差异：

  1. _pcg_fwd_count 仅在 graph replay 时递增（capture、warmup 不计），而
  _eager_fwd_count 在所有 extend forward 中递增（包含 startup warmup）。两者都设
  PROFILE_FWD=10，但 fwd#10 对应不同的 serving forward 序号。
  2. PCG fwd#10 采集到 chunk2 (seq_kv=8192, Q 被 pad 到 8192, FA≈2230us)，Eager
  fwd#10 采集到 chunk1 (seq_kv=4096, FA≈627us)。比值 2230/627=3.56x 与 O(seq²)
  缩放完全吻合。
  3. 独立微基准验证两块 NPU 上 FA 性能完全相同（chunk1: ~712us vs
  ~738us，chunk2: ~2291us vs ~2350us）。

  修正后的 PCG 实际开销： 原始 50.1ms delta 中，~16ms 是伪 FA
  delta（排除），~20.5ms 是 AllReduce 并发 outlier，真实 PCG 固有开销约 13.5ms。

  下一步： 如果需要准确的 profiler 对比，需要确保 PCG 和 Eager 采集同一 chunk
  类型（调整 PROFILE_FWD 编号补偿偏移量），或者直接在 profiler 代码中加入 chunk
  类型过滤逻辑。

❯ 下一步应该怎么做                                                              

  Thought for 11s (ctrl+o to expand)

● 当前状态：FA 伪影已排除，PCG 的真实瓶颈清晰了——replay 超线性 scaling（shape
  2048→4096 时 1.59x，吃掉全部编译收益）。有几个方向：

  1. 修 profiler 计数器偏移 — 在 profiler 代码里按 chunk 类型过滤或对齐 fwd
  编号，确保未来 profiler 对比是 apples-to-apples。工作量小，防止再踩坑。
  2. 用对齐后的 profiler 做干净的 PCG vs Eager 对比 — 拿到同一 chunk
  类型下的完整 kernel 级拆分，定量看 13.5ms 真实开销花在哪里（replay
  dispatch、stream sync、还是 CANN runtime）。
  3. 分析 replay 超线性 scaling 根因 — 这是决定 PCG 在 shape=4096
  能否获益的核心问题。需要 CANN profiler 看 41 个 NPUGraph.replay 的 device
  kernel 分布。
  4. 清理插桩代码 — 把 ascend_torch_native_backend.py 里加的 shape log 去掉。

  建议路径：先做 1（小改动），再做 2 拿干净数据，然后 3 定位 replay 瓶颈。第 4
  步随时可以做。你想先推哪个方向？

❯ 可以，先做1这个小改动                                                         

  Thought for 57s (ctrl+o to expand)

● Let me look at how the forward context is accessible in the PCG profiler path,
  so I can add chunk type info to both profilers.

  Searched for 3 patterns, read 1 file (ctrl+o to expand)

● get_tc_piecewise_forward_context() gives access to forward_batch during
  replay. Now let me read the full PCG profiler code to plan the change.

  Read 1 file (ctrl+o to expand)
● Now I have the full picture. Let me implement the fix — add chunk type info to
  both profilers' log messages and directory names by extracting
  extend_prefix_lens from the forward batch.

● Update(python/sglang/srt/compilation/npu_piecewise_backend.py)
  ⎿  Added 3 lines
      16      CUDAPiecewiseBackend,
      17      weak_ref_tensors,
      18  )
      19 +from sglang.srt.model_executor.runner_backend_utils.tc_piecewis
         +e_cuda_graph import (                                          
      20 +    get_tc_piecewise_forward_context,                          
      21 +)                                                              
      22  
      23  # SGLANG_NPU_PROFILE_FWD=N  — profile forward #N with torch_npu
          .profiler
      24  # SGLANG_NPU_PROFILE_DIR=path — output directory (default: /tmp
          /npu_profile)

● Now add a helper to extract chunk type info from the forward context, and use
  it in the profiler log:

● Update(python/sglang/srt/compilation/npu_piecewise_backend.py)
  ⎿  Added 10 lines, removed 2 lines
      139                  if _pcg_fwd_count == _PROFILE_FWD:
      140                      import torch_npu.profiler as npu_prof
      141  
      142 +                    chunk_tag = ""                            
      143 +                    ctx = get_tc_piecewise_forward_context()  
      144 +                    if ctx and ctx.forward_batch is not None: 
      145 +                        fb = ctx.forward_batch                
      146 +                        prefix = int(fb.extend_prefix_lens[0].
          +item()) if fb.extend_prefix_lens is not None and len(fb.extend
          +_prefix_lens) > 0 else -1                                     
      147 +                        seq_kv = int(fb.seq_lens[0].item()) if
          + fb.seq_lens is not None and len(fb.seq_lens) > 0 else -1     
      148 +                        chunk_tag = f"_prefix{prefix}_seqkv{se
          +q_kv}"                                                        
      149 +                                                              
      150                      out_dir = os.path.join(
      143 -                        _PROFILE_DIR, f"pcg_fwd{_pcg_fwd_count
          -}_shape{runtime_shape}"                                       
      151 +                        _PROFILE_DIR, f"pcg_fwd{_pcg_fwd_count
          +}_shape{runtime_shape}{chunk_tag}"                            
      152                      )
      153                      os.makedirs(out_dir, exist_ok=True)
      154                      _profile_ctx = npu_prof.profile(
     ...
      161                      _profile_ctx.__enter__()
      162                      print(
      163                          f"[PCG Profile] START fwd#{_pcg_fwd_co
           unt} "
      156 -                        f"shape={runtime_shape} -> {out_dir}" 
      164 +                        f"shape={runtime_shape}{chunk_tag} -> 
          +{out_dir}"                                                    
      165                      )
      166  
      167          entry.cudagraph.replay()

● Now do the same for the Eager profiler:

● Update(python/sglang/srt/model_executor/runner/eager_runner.py)
  ⎿  Added 7 lines, removed 2 lines
      356                          import torch_npu.profiler as npu_prof
      357  
      358                          shape = len(forward_batch.input_ids)
      359 +                        chunk_tag = ""                        
      360 +                        if forward_batch.extend_prefix_lens is
          + not None and len(forward_batch.extend_prefix_lens) > 0:      
      361 +                            prefix = int(forward_batch.extend_
          +prefix_lens[0].item())                                        
      362 +                            seq_kv = int(forward_batch.seq_len
          +s[0].item()) if forward_batch.seq_lens is not None and len(for
          +ward_batch.seq_lens) > 0 else -1                              
      363 +                            chunk_tag = f"_prefix{prefix}_seqk
          +v{seq_kv}"                                                    
      364                          out_dir = os.path.join(
      365                              _EAGER_PROFILE_DIR,
      361 -                            f"eager_fwd{_eager_fwd_count}_shap
          -e{shape}",                                                    
      366 +                            f"eager_fwd{_eager_fwd_count}_shap
          +e{shape}{chunk_tag}",                                         
      367                          )
      368                          os.makedirs(out_dir, exist_ok=True)
      369                          print(
      370                              f"[Eager Profile] START fwd#{_eage
           r_fwd_count} "
      366 -                            f"shape={shape} -> {out_dir}"     
      371 +                            f"shape={shape}{chunk_tag} -> {out
          +_dir}"                                                        
      372                          )
      373                          with npu_prof.profile(
      374                              activities=[

● Let me verify the final state of both files.

  Read 2 files (ctrl+o to expand)

● 改动完成。现在两个 profiler 的日志和输出目录名都会包含 chunk 类型信息：

  之前：
  [PCG Profile] START fwd#10 shape=4096 -> .../pcg_fwd10_shape4096
  [Eager Profile] START fwd#10 shape=4096 -> .../eager_fwd10_shape4096

  之后：
  [PCG Profile] START fwd#10 shape=4096_prefix4096_seqkv8192 ->
  .../pcg_fwd10_shape4096_prefix4096_seqkv8192
  [Eager Profile] START fwd#10 shape=4096_prefix0_seqkv4096 ->
  .../eager_fwd10_shape4096_prefix0_seqkv4096

  这样一眼就能看出两个 profiler 是否采集了同一个 chunk 类型：
  - prefix0 = chunk1（首 chunk，无前缀）
  - prefix>0 = chunk2（有前缀的续 chunk）
  - seqkv = 实际 KV 长度，直接决定 FA kernel 耗时

❯ 各对 pcg和eager进行一次4096 的 bench serving 吗                               

  Thought for 8s (ctrl+o to expand)

● Let me check the startup scripts first.

  Read 1 file, listed 2 directories (ctrl+o to expand)

● 脚本都在。需要你手动执行以下步骤：

  1. 启动两个服务器（两个终端）：

  # 终端 1 — PCG ON（port 10010）
  SGLANG_NPU_PROFILE_FWD=10
  SGLANG_NPU_PROFILE_DIR=/home/hanwlax/workspace/logs/npu_profile_chunk_fix
  ./pcg_on.sh

  # 终端 2 — Eager（port 10086）
  SGLANG_NPU_PROFILE_FWD=10
  SGLANG_NPU_PROFILE_DIR=/home/hanwlax/workspace/logs/npu_profile_chunk_fix
  ./pcg_off.sh

  2. 两个服务器都 ready 后，跑 bench serving：

  cd /home/hanwlax/workspace/progress/pcg
  ./bench_serving.sh all -t 8192 -p 20 -m 1

  这次 profiler 输出目录和日志会带上 chunk 类型信息，例如：
  [PCG Profile] START fwd#10 shape=4096_prefix4096_seqkv8192 ->
  .../pcg_fwd10_shape4096_prefix4096_seqkv8192
  [Eager Profile] START fwd#10 shape=4096_prefix0_seqkv4096 ->
  .../eager_fwd10_shape4096_prefix0_seqkv4096

  这样你一眼就能确认两个 profiler 是否采到了同一 chunk 类型。如果不一致，调整
  PROFILE_FWD 的值（奇偶加减 1）使它们对齐。

❯ pcg_on 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_114525.jsonl                                        
  pcg_on 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_114525.log                                          
  pcg_off 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_114525.jsonl                                      
  pcg_off 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_114525.log                                        
  服务日志：eager: sglang_run_pcg_off_20260625_193413.log                       
  pcg: sglang_run_pcg_on_20260625_193416.log                                    

  Searched for 2 patterns (ctrl+o to expand)

● chunk 类型标签生效了，而且正好验证了之前的结论：

  - PCG fwd#10: prefix4096_seqkv8192 → chunk2（FA≈2230us）
  - Eager fwd#10: prefix0_seqkv4096 → chunk1（FA≈627us）

  确认 fwd#10 对应的是不同 chunk 类型。要做 apples-to-apples 对比，需要调整
  PROFILE_FWD 让两者都采到同一 chunk。Eager fwd#10 是 chunk1，fwd#11 应该是
  chunk2；PCG fwd#10 是 chunk2，fwd#11 应该是 chunk1。所以：

  - 同采 chunk1: PCG 用 PROFILE_FWD=11，Eager 用 PROFILE_FWD=10
  - 同采 chunk2: PCG 用 PROFILE_FWD=10，Eager 用 PROFILE_FWD=11

  要不要现在用对齐的编号重跑一次，拿到干净的同 chunk 对比数据？

❯ 可以，怎么                                                                    

  Thought for 8s (ctrl+o to expand)

● 重新启动两个服务器，用不同的 PROFILE_FWD 让它们都采到 chunk1（prefix=0,
  seqkv=4096）：

  终端 1 — PCG ON（用 fwd#11）：
  SGLANG_NPU_PROFILE_FWD=11
  SGLANG_NPU_PROFILE_DIR=/home/hanwlax/workspace/logs/npu_profile_aligned
  ./pcg_on.sh

  终端 2 — Eager（用 fwd#10）：
  SGLANG_NPU_PROFILE_FWD=10
  SGLANG_NPU_PROFILE_DIR=/home/hanwlax/workspace/logs/npu_profile_aligned
  ./pcg_off.sh

  两个都 ready 后跑 bench：
  cd /home/hanwlax/workspace/progress/pcg
  ./bench_serving.sh all -t 8192 -p 20 -m 1

  跑完后看日志确认两者都显示 prefix0_seqkv4096（chunk1），就是对齐的了。

❯ eager: /home/hanwlax/workspace/logs/server_log/sglang_run_pcg_off_20260625_19 
  5109.log                                                                      
  pcg:                                                                          
  /home/hanwlax/workspace/logs/server_log/sglang_run_pcg_on_20260625_195109.log 
  pcg_on 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_115546.jsonl                                        
  pcg_on 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_115546.log                                          
  pcg_off 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_115546.jsonl                                      
  pcg_off 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_115546.log                                        

  Searched for 1 pattern (ctrl+o to expand)

● 两者都是 prefix0_seqkv4096（chunk1），对齐了。现在来分析 kernel 级数据。

  Searched for 1 pattern, ran 1 shell command (ctrl+o to expand)

● FA 完全对齐了！PCG avg≈633.9us vs Eager avg≈635.7us — 几乎一样。现在来看整体
  kernel 开销对比。

  Thought for 33s, ran 2 shell commands (ctrl+o to expand)

● Wait — let me re-check the CSV format. Column 1 might be device_id, not count.

  Ran 1 shell command (ctrl+o to expand)

● Column 1 是 Device_id，不是 count。每行是一次独立的 kernel 调用。重新分析：

  Ran 2 shell commands (ctrl+o to expand)

● 非常清晰的对比结果！同 chunk 对齐后，关键发现：

  1. 大部分 kernel 时间几乎一致：

  ┌───────────────────┬───────────────┬───────────────┬───────┐
  │    Kernel 类别    │      PCG      │     Eager     │ 差异  │
  ├───────────────────┼───────────────┼───────────────┼───────┤
  │ FlashAttention    │ 6.34ms (10x)  │ 6.36ms (10x)  │ 0%    │
  ├───────────────────┼───────────────┼───────────────┼───────┤
  │ MoE mega_kernel   │ 30.53ms (30x) │ 30.85ms (30x) │ -1%   │
  ├───────────────────┼───────────────┼───────────────┼───────┤
  │ GroupedMatmul     │ 39.10ms (80x) │ 38.75ms (80x) │ +0.9% │
  ├───────────────────┼───────────────┼───────────────┼───────┤
  │ MatMulV3          │ 22.89ms (80x) │ 22.80ms (80x) │ +0.4% │
  ├───────────────────┼───────────────┼───────────────┼───────┤
  │ IndexCopy/Scatter │ 20.13ms (60x) │ 20.44ms (60x) │ -1.5% │
  └───────────────────┴───────────────┴───────────────┴───────┘

  2. AllReduce 是唯一显著差异：
  - PCG: hcom_allReduce 80x, total=12.56ms, avg=157us
  - Eager: hcom_allReduce 81x (多 1 次), total=30.57ms, avg=377us
  - Eager 有一个 17.2ms 的 AllReduce 
  outlier（hcom_allReduce__503_0_1），单次就占了整个差额

  3. Device 总时间：
  - PCG: 260.4ms
  - Eager: 274.7ms
  - Delta: -14.3ms，几乎全来自 AllReduce 差异 (30.57 - 12.56 = 18ms)

  结论：在 device kernel 层面，PCG 并没有额外开销。shape=4096 时 PCG 与 Eager
  持平（bench serving -0.7%~-1.3%）的原因不是 PCG 引入了开销，而是 graph replay
  的 host-side dispatch 开销（~3.9ms）被 AllReduce 的波动掩盖了。AllReduce
  outlier 是并发/timing 相关的，非 PCG 固有问题。

❯ kernel launch省下来的时间被那一部分吃掉了呢                                   
  Thought for 39s (ctrl+o to expand)

● 好问题。kernel duration 总和只是 device 忙碌时间，不包括 kernel
  之间的空隙（device 等待 host 发射下一个 kernel）。PCG
  理论上省的就是这些空隙。让我看 operator 层面的 wall-clock 时间线。

  Thought for 13s, searched for 1 pattern, ran 2 shell commands (ctrl+o to expand)

● 关键发现！PCG host-side 比 Eager 多了 66.8ms。让我深入看
  _local_scalar_dense（.item() 调用，会触发 D2H 同步）。

  Ran 2 shell commands (ctrl+o to expand)

● 找到根因了。

  Ran 1 shell command (ctrl+o to expand)

● 根因清楚了。PCG 省下的 kernel launch 时间被 .item() 同步开销吃掉了：

  Host-side 开销对比：

  ┌────────────────────────────────┬──────────────┬──────────────┬─────────┐
  │              类别              │     PCG      │    Eager     │  Delta  │
  ├────────────────────────────────┼──────────────┼──────────────┼─────────┤
  │ _local_scalar_dense (.item())  │ 189.7ms      │ 115.7ms      │ +74.0ms │
  │                                │ (310x)       │ (515x)       │         │
  ├────────────────────────────────┼──────────────┼──────────────┼─────────┤
  │ unified_*_attention (split     │ 45.0ms (40x) │ 0ms          │ +45.0ms │
  │ point)                         │              │              │         │
  ├────────────────────────────────┼──────────────┼──────────────┼─────────┤
  │ 其他 host ops                  │ 73.4ms       │ 125.6ms      │ -52.2ms │
  ├────────────────────────────────┼──────────────┼──────────────┼─────────┤
  │ Host 总计                      │ 308.1ms      │ 241.3ms      │ +66.8ms │
  └────────────────────────────────┴──────────────┴──────────────┴─────────┘

  核心问题： PCG 的 .item() 调用次数虽然更少（310 vs 515），但每次更贵：
  - PCG: 39 次 ≥ 0.5ms，其中 27 次在 3-4ms，12 次在 4ms+
  - Eager: 69 次 ≥ 0.5ms，主要在 1-2ms 范围

  原因： PCG 模式下，graph replay 把大量 kernel 快速排到 device queue，当
  attention split point 的 .item() 触发 D2H 同步时，需要等整个子图的 kernel
  全部执行完才能拿到值。而 Eager 模式下 kernel 逐个发射，.item() 时 device queue
  不深，等待时间短。

  这就是 kernel launch 省下的时间被吃掉的机制： graph replay 加速了 kernel 发射
  → device queue 更深 → split point 的 .item() 同步等待更长 → 抵消了发射加速。

  这 310 个 .item() 调用来自 run_sdpa_forward_extend 中
  seq_lens[i].item()、extend_prefix_lens[i].item() 等。如果能把这些值在 replay
  开始前预先拷到 host（或用 CPU tensor 传递），就能消除这个瓶颈。

❯ 试一下                                                                        

  Thought for 17s (ctrl+o to expand)

● Let me trace the .item() calls in the attention split point to find where to
  pre-copy values to host.

  Searched for 2 patterns (ctrl+o to expand)

● Now let me check how forward_batch is populated for replay — which of these
  fields already have CPU copies.

  Searched for 1 pattern (ctrl+o to expand)

● forward_batch 已经有 seq_lens_cpu、extend_prefix_lens_cpu、extend_seq_lens_cpu
  字段。让我看 PCG replay 时这些是否被设置。

  Searched for 1 pattern (ctrl+o to expand)

● CPU 副本在 replay 时已经传了。现在看 run_sdpa_forward_extend 为什么没用它们 —
  它接收的是 device tensor 参数。

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● 看到了 — 传的是 forward_batch.seq_lens（device tensor），而不是
  forward_batch.seq_lens_cpu。让我看另一个调用点。

  Read 1 file (ctrl+o to expand)

● 方案明确了：修改 run_sdpa_forward_extend 接受可选的 CPU 值，避免 .item()
  同步。先改 SDPA 后端，然后在调用侧传入 CPU 值。

  Read 1 file (ctrl+o to expand)

● Now let me also check run_sdpa_forward_decode for .item() calls.

  Read 1 file (ctrl+o to expand)

● 现在修改 run_sdpa_forward_extend，接受可选的 CPU 列表参数，避免 .item() D2H
  同步。

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_ba
        ckend.py)
  ⎿  Added 1 line, removed 1 line
      2  
      3  import math
      4  import os
      5 -from typing import Optional                                     
      5 +from typing import List, Optional, Union                        
      6  
      7  import torch
      8  from torch.nn.functional import scaled_dot_product_attention

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_ba
        ckend.py)
  ⎿  Added 15 lines, removed 5 lines
      80          full_to_swa_mapping: Optional[torch.Tensor] = None,
      81          logit_cap: float = 0.0,
      82          logit_capping_method: str = "tanh",
      83 +        seq_lens_cpu: Optional[Union[List[int], torch.Tensor]] 
         += None,                                                        
      84 +        extend_prefix_lens_cpu: Optional[Union[List[int], torch
         +.Tensor]] = None,                                              
      85 +        extend_seq_lens_cpu: Optional[Union[List[int], torch.Te
         +nsor]] = None,                                                 
      86      ):
      87          """Run the extend forward by using torch native sdpa op
          .
      88  
     ...
      104              sliding_window_size: int, -1 means no sliding wind
           ow
      105              full_to_swa_mapping: mapping from full pool index 
           to SWA pool index,
      106                  required for SWA layers to translate req_to_to
           ken indices
      107 +            seq_lens_cpu: pre-copied host values to avoid D2H 
          +sync                                                          
      108 +            extend_prefix_lens_cpu: pre-copied host values to 
          +avoid D2H sync                                                
      109 +            extend_seq_lens_cpu: pre-copied host values to avo
          +id D2H sync                                                   
      110  
      111          Returns:
      112              output: [num_tokens, num_heads, head_size]
     ...
      115          assert seq_lens.shape[0] == extend_prefix_lens.shape[0
           ]
      116          assert seq_lens.shape[0] == extend_seq_lens.shape[0]
      117  
      118 +        def _cpu_val(cpu_list, device_tensor, idx):           
      119 +            if cpu_list is not None:                          
      120 +                v = cpu_list[idx]                             
      121 +                return int(v.item()) if isinstance(v, torch.Te
          +nsor) else int(v)                                             
      122 +            return int(device_tensor[idx].item())             
      123 +                                                              
      124          # [num_tokens, num_heads, head_size] -> [num_heads, nu
           m_tokens, head_size]
      125          query = query.movedim(0, query.dim() - 2)
      126  
      127          start_q, start_kv = 0, 0
      128          for seq_idx in range(seq_lens.shape[0]):
      117 -            # Need optimize the performance later.            
      129 +            extend_seq_len_q = _cpu_val(extend_seq_lens_cpu, e
          +xtend_seq_lens, seq_idx)                                      
      130 +            prefill_seq_len_q = _cpu_val(extend_prefix_lens_cp
          +u, extend_prefix_lens, seq_idx)                               
      131  
      119 -            extend_seq_len_q = int(extend_seq_lens[seq_idx].it
          -em())                                                         
      120 -            prefill_seq_len_q = int(extend_prefix_lens[seq_idx
          -].item())                                                     
      121 -                                                              
      122 -            seq_len_kv = int(seq_lens[seq_idx].item())        
      132 +            seq_len_kv = _cpu_val(seq_lens_cpu, seq_lens, seq_
          +idx)                                                          
      133              end_q = start_q + extend_seq_len_q
      134              end_kv = start_kv + seq_len_kv
      135              atten_start_kv = 0

● Now update the call sites in ascend_backend.py to pass CPU values:

  Read 1 file (ctrl+o to expand)

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py)
  ⎿  Added 3 lines
      1527                          ),
      1528                          logit_cap=layer.logit_cap,
      1529                          logit_capping_method=layer.logit_capp
            ing_method,
      1530 +                        seq_lens_cpu=forward_batch.seq_lens_c
           +pu,                                                          
      1531 +                        extend_prefix_lens_cpu=forward_batch.
           +extend_prefix_lens_cpu,                                      
      1532 +                        extend_seq_lens_cpu=forward_batch.ext
           +end_seq_lens_cpu,                                            
      1533                      )
      1534                      attn_output = attn_output.view(
      1535                          -1, layer.tp_q_head_num * layer.v_hea
            d_dim

● Now the second call site:

  Read 1 file (ctrl+o to expand)

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py)
  ⎿  Added 3 lines
      1765                      scaling=layer.scaling,
      1766                      enable_gqa=use_gqa,
      1767                      causal=True,
      1768 +                    seq_lens_cpu=forward_batch.seq_lens_cpu, 
      1769 +                    extend_prefix_lens_cpu=forward_batch.exte
           +nd_prefix_lens_cpu,                                          
      1770 +                    extend_seq_lens_cpu=forward_batch.extend_
           +seq_lens_cpu,                                                
      1771                  )
      1772              else:
      1773                  num_token_padding = q.shape[0]

● Now let me also check the other .item() calls in ascend_backend.py — lines 772
  and 786 which also do per-request .item():

  Read 1 file (ctrl+o to expand)
● This is a different code path (block-based attention, not the SDPA path we're
  focusing on). The profiler showed the hot .item() calls are in the SDPA path.
  Let me also check line 658 and 1422:

  Read 1 file (ctrl+o to expand)

● That's a single .item() call on a different path. The 310 .item() calls in PCG
  profiler are mostly from the per-request loop in run_sdpa_forward_extend (3
  .item() calls × ~10 attention layers × ~10 requests ≈ 300+). Let me also clean
  up the shape logging we added earlier, and verify the code compiles.

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_ba
        ckend.py)
  ⎿  Removed 18 lines
      198                      .movedim(query.dim() - 2, 0)
      199                  )
      200              else:
      201 -                _q_in = per_req_query_redundant.unsqueeze(0)  
      202 -                _k_in = per_req_key.unsqueeze(0)              
      203 -                _v_in = per_req_value.unsqueeze(0)            
      204 -                if int(os.environ.get("SGLANG_NPU_SDPA_SHAPE_L
          -OG", "0")):                                                   
      205 -                    _fmt = getattr(torch_npu, 'get_npu_format'
          -, None)                                                       
      206 -                    _q_fmt = _fmt(_q_in) if _fmt else 'N/A'   
      207 -                    _k_fmt = _fmt(_k_in) if _fmt else 'N/A'   
      208 -                    _v_fmt = _fmt(_v_in) if _fmt else 'N/A'   
      209 -                    print(                                    
      210 -                        f"[SDPA] seq_idx={seq_idx} Q={list(_q_
          -in.shape)} K={list(_k_in.shape)} V={list(_v_in.shape)}"       
      211 -                        f" Q.strides={_q_in.stride()} K.stride
          -s={_k_in.stride()}"                                           
      212 -                        f" Q.contig={_q_in.is_contiguous()} K.
          -contig={_k_in.is_contiguous()}"                               
      213 -                        f" Q.fmt={_q_fmt} K.fmt={_k_fmt} V.fmt
          -={_v_fmt}"                                                    
      214 -                        f" Q.data_ptr={_q_in.data_ptr():#x}"  
      215 -                        f" extend_seq_len_q={extend_seq_len_q}
          - prefix_len={prefill_seq_len_q}"                              
      216 -                        f" seq_len_kv={seq_len_kv} causal={cau
          -sal} gqa={enable_gqa}",                                       
      217 -                        flush=True,                           
      218 -                    )                                         
      201                  per_req_out_redundant = (
      202                      scaled_dot_product_attention(
      203                          _q_in,

● Now clean up unused imports that were added for shape logging:

  Read 1 file (ctrl+o to expand)

● os and torch_npu are no longer needed after removing the shape log. Let me
  check if they're used elsewhere in the file.

  Searched for 1 pattern (ctrl+o to expand)

● No longer used. Clean them up:

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_ba
        ckend.py)
  ⎿  Removed 6 lines
       1  from __future__ import annotations
       2  
       3  import math
       4 -import os                                                      
       4  from typing import List, Optional, Union
       5  
       6  import torch
       7  from torch.nn.functional import scaled_dot_product_attention
       8  
      10 -try:                                                           
      11 -    import torch_npu                                           
      12 -except ImportError:                                            
      13 -    torch_npu = None                                           
       9  
      15 -                                                               
      10  class AscendTorchNativeAttnBackend:
      11      def __init__(self):
      12          pass

● Let me verify the final state of the modified file:

  Read 1 file (ctrl+o to expand)

● Good. Now let me also check how seq_lens_cpu is structured — it might be a
  torch.Tensor on CPU, so indexing with [idx] gives a tensor element that still
  needs .item(). The _cpu_val helper handles this. Let me verify the types:

  Searched for 2 patterns (ctrl+o to expand)

● - seq_lens_cpu: torch.Tensor (CPU) — indexing gives tensor, needs .item()
  - extend_prefix_lens_cpu: List[int] — indexing gives int directly
  - extend_seq_lens_cpu: List[int] — indexing gives int directly

  The _cpu_val helper handles both cases correctly. Now let me also check that
  in the PCG replay path, seq_lens_cpu is properly set as a CPU tensor:

  Read 1 file (ctrl+o to expand)

● All three CPU fields (seq_lens_cpu, extend_prefix_lens_cpu,
  extend_seq_lens_cpu) are passed through from the original batch to the static
  forward batch. The change should work.

  Let me do a quick syntax check:

  Ran 2 shell commands (ctrl+o to expand)

● 改动完成。总结一下修改：

  修改了 3 个文件：

  1. ascend_torch_native_backend.py — run_sdpa_forward_extend 新增
  seq_lens_cpu、extend_prefix_lens_cpu、extend_seq_lens_cpu 可选参数，优先从 CPU
  值读取，避免 .item() 触发 D2H 同步。清理了之前的 shape log 插桩。
  2. ascend_backend.py — 两个 run_sdpa_forward_extend 调用点传入 forward_batch
  上已有的 CPU 副本。
  3. npu_piecewise_backend.py — 之前改的 profiler chunk 标签（已完成）。

  这样每个 attention layer 的 SDPA 路径就不会再做 D2H 同步了。之前 PCG 模式下
  310 次 .item() 共花 189.7ms（vs Eager 115.7ms），消除后理论上可以省 74ms+。

  可以重新启动服务器跑 bench serving
  验证效果。启动命令和之前一样，不需要加额外环境变量。

❯ 我使用 SGLANG_NPU_PROFILE_FWD=11                                              
  SGLANG_NPU_PROFILE_DIR=/home/hanwlax/workspace/logs/npu_profile_item_fix      
  ./pcg_off.sh启动了on/off服务                                                  

  Thought for 14s (ctrl+o to expand)

● 注意 Eager 用 PROFILE_FWD=11 会采到 chunk2（之前验证过 Eager
  fwd#10=chunk1，fwd#11=chunk2），不过这不影响 bench serving 的吞吐量对比。

  等两个服务都 ready 后直接跑：

  cd /home/hanwlax/workspace/progress/pcg
  ./bench_serving.sh all -t 8192 -p 20 -m 1

  跑完把结果路径和服务日志发我，我来对比吞吐量和 profiler 中 .item()
  的开销变化。

❯ 报错了 sglang_run_pcg_off_20260625_201659.log                                 

  Searched for 1 pattern (ctrl+o to expand)

● _q_in is not defined — 删掉 shape log 时遗留了引用。让我看具体位置。

  Read 1 file (ctrl+o to expand)

● The shape log removal left _q_in, _k_in, _v_in references but removed their
  definitions. Need to revert to using the original .unsqueeze(0) directly.

● Update(python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_ba
        ckend.py)
  ⎿  Added 3 lines, removed 3 lines
      194              else:
      195                  per_req_out_redundant = (
      196                      scaled_dot_product_attention(
      197 -                        _q_in,                                
      198 -                        _k_in,                                
      199 -                        _v_in,                                
      197 +                        per_req_query_redundant.unsqueeze(0), 
      198 +                        per_req_key.unsqueeze(0),             
      199 +                        per_req_value.unsqueeze(0),           
      200                          enable_gqa=enable_gqa,
      201                          scale=scaling,
      202                          is_causal=causal,

  Ran 1 shell command (ctrl+o to expand)

● 修好了，之前删 shape log 时漏了把 _q_in/_k_in/_v_in 改回原始的 .unsqueeze(0)
  形式。重启服务器即可。

❯ /home/hanwlax/workspace/logs/server_log/sglang_run_pcg_on_20260625_202802.log 
  /home/hanwlax/workspace/logs/server_log/sglang_run_pcg_off_20260625_202752.lo 
  g                                                                             
  pcg_on 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_124645.jsonl                                        
  pcg_on 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/pc 
  g_on-8192-20-0-1-20260625_124645.log                                          
  pcg_off 结果文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_124645.jsonl                                      
  pcg_off 日志文件: /home/hanwlax/workspace/progress/pcg/bench_serving_result/p 
  cg_off-8192-20-0-1-20260625_124645.log                                        
                                                                                
  为什么bench serving结果差这么多                                               

  Searched for 3 patterns, read 2 files (ctrl+o to expand)

● 没有报错，profiler 正常跑完。问题是 profiler 本身拖慢了 serving。P99 TTFT
  差距大是因为 fwd#11 的 profiler 采集发生在请求处理中间，PCG profiler 采到的是
  chunk1 (prefix0_seqkv4096)，Eager 采到的是 chunk2 (prefix4096_seqkv8192) —
  chunk2 的 profiler trace 更大、写盘更慢。

  这次 bench serving 结果 (0.63 vs 0.88 req/s) 不能反映 .item() 优化效果，因为
  profiler 干扰了。

  建议不开 profiler 重跑一次干净的 bench serving 来看纯吞吐量：

  # 终端 1 — PCG ON (不开 profiler)
  ./pcg_on.sh

  # 终端 2 — Eager (不开 profiler)
  ./pcg_off.sh

  两个都 ready 后：
  ./bench_serving.sh all -t 8192 -p 20 -m 1

  如果想对比 profiler 中 .item() 的变化，可以之后单独开一次 profiler 跑。

❯ /rename   ⎿  Session renamed to: bench-serving-profiler-interference

❯ /resume 
  ⎿  Resume cancelled