# DualStream 四卡训练实际内核与瓶颈（2026-11-22）

本次实测基于 `dce08d46ba37423dbfdc89eba950906225967dbd` 的 DualStream、
真实 LIBERO 数据、每卡 batch64、global256、GC 开启、ZeRO-2，
`NCCL_P2P_LEVEL=SYS`。运行在常驻开发机 `di-dc17as3hq4us73eg9p7g`
（research 组，实际节点 pro6000-21），PyTorch 2.8.0+cu128、NCCL 2.27.3。

## 实际 attention 后端

使用 torch.profiler 捕获第7次 optimizer update 的 rank0 CPU/CUDA trace，
由模块范围标记、ATen 算子名与实际 Input type 交叉确认。不是仅检查 YAML
或 `flash_sdp_enabled()` 开关。

| 模块 | 实际后端 | Q/K/V dtype | 原始前向调用数 |
| --- | --- | --- | ---: |
| ViT | `aten::_scaled_dot_product_flash_attention` | BF16 | 24 |
| DualStream 跨视角融合 | `aten::_scaled_dot_product_flash_attention` | BF16 | 12 |
| LLM | `aten::_scaled_dot_product_efficient_attention` | BF16 | 28 |
| Action DiT | `aten::_scaled_dot_product_efficient_attention` | FP32 | 28 |

因此视觉与融合已经使用 PyTorch 内置 FlashAttention；LLM 和动作头使用
Memory-Efficient Attention。没有观察到 SDPA math fallback，也没有使用外部
flash-attn/FA4 adapter。LLM 输入包含显式 attention mask；动作头14次交叉注意力
带 mask、14次自注意力不带 mask，但 Q/K/V 都是 FP32。
这是当前输入与运行环境下的分派结果，不保证其他 shape/mask/dtype 使用同一内核。

GC 重算还触发了额外24次视觉、27次语言前向 attention，计入完整训练步耗时。

## 正常 step 时间与 profiler 分解

共完成12次更新。正常耗时取第5/6/9/10/11/12步中位数，排除采样步和导出后
紧接的一步，以免 profiler 开销及其他 rank 等待导出影响基线。
rank0 正常耗时2.714s，其他 rank 约2.728s；rank0 前向0.708s、反向1.656s、
engine step 0.340s。数据等待中位数0.723ms，allocated 显存峰值约51.06GiB。
采样步自身耗时2.784s。

采样步只统计 trace 的 `kernel`、`gpu_memcpy`、`gpu_memset` 事件，**排除
`gpu_user_annotation`**，避免把阶段/模块范围标记当作额外 kernel 重复计数。
共32,372个GPU操作（25,805个kernel），活跃时间合计2,556.46ms。
本次这些事件的时间并集与累加值一致，没有可见重叠。

| 分类 | GPU时间 | 占采样步GPU活跃时间 |
| --- | ---: | ---: |
| GEMM/矩阵乘法 | 1090.36ms | 42.65% |
| 其他计算：逐元素、归一化、索引、转换等 | 802.93ms | 31.41% |
| NCCL 通信 | 552.93ms | 21.63% |
| attention 内核（含反向与重算） | 77.07ms | 3.01% |
| memcpy | 31.94ms | 1.25% |
| memset | 1.24ms | 0.05% |

百分比是GPU操作活跃时间口径，不是端到端step占比；GEMM/其他计算是依据kernel
名称归类，attention 的 QKV/output projection 被归入矩阵乘法，不在77ms内。
CPU调度/同步等待、GPU空隙需要结合完整时间线看，不能把剩余时间全称为数据瓶颈。

## 已定位的热点与后续优先级

1. **动作头 FP32 计算。** Trace 中其203次前向 `aten::addmm` 输入均为 float32，
   GPU累计约213.14ms；动作头attention的Q/K/V也均为float32。
   完整step中出现大量 `cutlass_80_simt_sgemm` 和 `magma_sgemmEx<float>`。
   视觉与语言的线性算子输入为BF16，因此配置写BF16不等于整个网络都以BF16计算。
   `QwenPI.py` 动作路径显式使用 `autocast(dtype=torch.float32)`；这不会启用BF16
   autocast。优先对照动作头TF32矩阵乘法，再考虑局部BF16，保留敏感标签、噪声和
   loss计算的精度，并验证动作、梯度和收敛。此次没有更改精度策略。

2. **张量操作与重算。** 存在大量逐元素、dtype转换和索引操作；视觉attention
   每层重新构造分桶索引（arange/cat/index_select/index_copy），GC还会重复执行。
   可优先验证固定shape下复用索引、减少无效拷贝/转换，再做局部compile融合。
   GC导致视觉/语言前向重算，后续可测试选择性checkpoint；当前峰值显存不能证明
   完全关闭GC一定不会OOM。未把所有802.93ms都归因于索引或Python开销。

3. **剩余通信。** 两个 `ncclDevKernel_AllGather_RING_LL` 累计278.56ms；
   BF16梯度AllReduce约274.34ms，另有两个很小的AllReduce。
   P2P修复后仍有优化空间，可以对照大包NCCL协议/通道或ZeRO策略，但本次没有
   测这些方案。上一轮SYS+overlap+50M bucket对照没有额外收益，不能直接推荐。

单纯换attention内核不是当前首要方向：即使理想化消除这77ms，也只覆盖当前
GPU活跃时间的约3%，而且无法消除QKV/输出投影矩阵乘法。

## 产物与验证范围

原始产物：`/hdfs/share-data-1/users/hjz/Workspace/artifacts/dualstream-profile-20261122`。

- `rank0-step7.trace.json`：Chrome/Perfetto trace，约129MB。
- `summary.json`：分类耗时、各模块attention后端/shape/dtype、正常步骤统计。
- `events.json`：CPU/GPU事件及模块范围关联；其中GPU范围标记不可重复计入kernel。
- `operators.txt`：原始聚合表，包含范围标记，不能直接对所有行求和。
- `phases-rank*.jsonl`、`runtime.json`、`modules.json`：各rank计时、运行时配置和模块映射。
- `profile_train.py`、`launch.py`、`analyze.py`：采集与分析脚本。

四卡完整12步正常退出，loss/grad norm均有限，结束后四卡显存释放。
只对rank0的一个预热后step做了详细kernel采样，其他rank记录阶段计时；这是瓶颈
定位证据，不是所有batch/节点的长期统计。未修改正式训练、模型实现或训练精度。
重新采样请使用新目录以保留原始记录。
