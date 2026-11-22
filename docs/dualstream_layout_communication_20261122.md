# DualStream 索引、重算与通信优化（2026-11-22）

本次只处理 profile 中的第2、3项：视觉张量整理/checkpoint 重算与 NCCL 通信。
保持 action head FP32、模型结构、数据、batch64/GPU、学习率与 ZeRO-2 不变。

## Agent 交接入口

- 工作树：`/hdfs/share-data-1/users/hjz/Workspace/starVLA-dualstream`；分支：
  `experiment/dualstream-bottleneck`。本次代码、配置、测试与报告随本分支提交管理；
  实测时的源码快照另存于下述 artifacts，正在运行的训练使用独立冻结源码。
  不要把相邻的 `starVLA` 工作树当成本次修改位置。
- 总入口：[DualStream 分支说明](dualstream_bottleneck.md)。前置证据分别是
  [通信诊断](dualstream_communication_tuning.md)和[实际内核 profile](dualstream_profile_20261122.md)。
- 当前默认：optimized recipe 开启索引复用；视觉 checkpoint 为 null，继承全局 true。
  关闭视觉 checkpoint 是已验证的可选配置，不是已启用的默认值。
  SYS 是本节点已验证的启动环境设置，尚未写入通用启动器；Simple/8通道仅为实验候选，
  结果不足以推荐为默认。不能把这些环境变量描述为已在正式训练任务中部署。
- 源码位置：`starVLA/model/modules/vlm/qwen3_vl_dual/modeling_qwen3_vl.py`
  实现布局复用；`starVLA/model/modules/vlm/QWen3DualStream.py` 传递配置；
  `starVLA/model/framework/VLM4A/DualStreamQwenPI.py` 实现视觉 checkpoint 覆盖。
  recipe 位于 `examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero_optimized.yaml`；
  新测试为 `tests/test_dualstream_attention_layout.py`。
- 本轮只完成有界 benchmark，未启动长期训练、未修改已有正式作业；结束时四卡空闲。
  后续使用常驻实例前重新检查占用。新作业仍由平台自动调度，不按报告中的实际节点绑机。
- 工作树中已有的 LIBERO-plus `preflight.py` 与 `launchers/test_shared_preflight.py`
  改动不属于本次优化，整理提交时保留并单独辨认。
- artifacts 中 `final-source/`、`final-source-sha256.json` 与 `implementation.patch`
  冻结了实测结束时的实现，之后的文档交接补充不回写该历史快照。

## 实现

- `framework.qwenvl.reuse_vision_attention_layout: true`：一个视觉 forward 只生成
  一次长度分组和索引，24层及其 checkpoint replay 复用。等长图像直接 reshape，
  避免每层 Q/K/V 的 gather 和输出 scatter。变长图像仍按长度分桶，索引只属于
  当前 forward，不建立跨输入缓存。优化 recipe 已开启；其他 recipe 默认关闭。
- `framework.qwenvl.vision_gradient_checkpointing: null`：默认继承全局 checkpoint
  策略；显式 false 关闭视觉重算，语言模型仍继承原设置。该开关仅在 DualStream
  framework 生效，调用视觉模型自身的 enable/disable 接口。false 不代表冻结视觉参数。
  96 GiB 卡上 batch64 实测约增加29 GiB 峰值 allocated；改变尺寸/batch后重新评估。
- 通信：保持 `NCCL_P2P_LEVEL=SYS`，进一步对照 Simple 协议和至少8个通信通道。
  NCCL 日志核实集合通信从默认2通道变为8通道。没有改系统级环境或其他作业。

## 对照方法

同一 research 常驻实例 `di-dc17as3hq4us73eg9p7g`，平台自动分配 `pro6000-21`；
4×RTX PRO 6000 Blackwell 96 GiB；PyTorch2.8/cu128、NCCL2.27.3、DeepSpeed0.17.6。
分支 `experiment/dualstream-bottleneck`，基于 dce08d46 的未提交改动。
使用真实 LIBERO 数据与 optimized recipe，每组20次 optimizer update，前5次预热，
后15次取中位数，global batch256，梯度累积1，种子42。各组顺序运行，无并发 GPU
benchmark。总步耗时包含取数据、前向、反向和 optimizer step；不含初始化、保存和
SwanLab 上传。完整 step 后同步；前反向之间不插同步。不是长期收敛试验。

## 真实训练结果与选择

| 配置 | 秒/步（中位数） | samples/s | 峰值 allocated GiB |
| --- | ---: | ---: | ---: |
| 基线：SYS，全局checkpoint | 2.684 | 95.38 | 51.06 |
| 索引复用 | 2.612 | 98.02 | 51.06 |
| 索引复用 + 关闭视觉checkpoint | 2.496 | 102.57 | 79.98 |
| 索引复用 + Simple/8通道 | 2.576 | 99.38 | 51.06 |
| 索引复用 + 关闭视觉checkpoint + Simple/8通道 | 2.534 | 101.01 | 79.98 |

推荐先使用索引复用：耗时减少2.7%，显存不增加。当前96 GiB卡、batch64若选择
关闭视觉checkpoint，可进一步达到总耗时减少7.0%、吞吐增加7.5%，代价约29 GiB。
该开关默认仍为null，让其他负载继续使用原有显存策略。

通信参数Simple/8通道在保留视觉checkpoint时仅比索引复用快1.4%；与关闭视觉
checkpoint组合后反而从2.496s变成2.534s，收益未稳定叠加。因此维持SYS作为明确
有效的设置，Simple/8通道暂不推荐为默认。仅凭一次15步稳态窗口不能保证小幅收益。

rank0的forward/backward/engine-step独立中位数（ms，不能直接视为纯计算/纯通信）：

| 配置 | forward | backward | engine step |
| --- | ---: | ---: | ---: |
| 基线：SYS，全局checkpoint | 695.0 | 1641.9 | 337.3 |
| 索引复用 | 667.8 | 1598.8 | 335.7 |
| 索引复用 + 关闭视觉checkpoint | 662.2 | 1480.4 | 336.3 |
| 索引复用 + Simple/8通道 | 668.9 | 1598.4 | 232.6 |
| 索引复用 + 关闭视觉checkpoint + Simple/8通道 | 662.8 | 1570.1 | 269.3 |

最后一组通过新增的YAML开关构造模型，验证正式配置入口；先前单独关闭视觉重算组通过benchmark包装器设置同样的视觉checkpoint状态。所有组均完成20步。

## 通信微基准

BF16 1 GiB，8次循环，去前3次，按各 rank 最大耗时求中位数，全部校验结果正确。

| 配置（均为SYS） | AllReduce ms | AllGather ms | ReduceScatter ms |
| --- | ---: | ---: | ---: |
| 自动协议/通道 | 38.46 | 20.62 | 22.18 |
| Simple | 38.28 | 20.62 | 22.30 |
| Simple + 至少8通道 | 37.11 | 18.91 | 18.85 |

Simple 单独未产生明显收益；联合配置的训练收益不能全部归因于协议变化。
这些集合通信小实验不等于实际训练提速，上述训练对照才是采用依据。
此前 SYS + overlap_comm + 50M buckets 实测2.703s，未优于仅SYS的2.687s，
因此没有直接开启该组合。参见 [此前通信诊断](dualstream_communication_tuning.md)。
NCCL 协议/通道参数参考 [NCCL 2.27 官方文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2275/user-guide/docs/env.html#nccl-proto)。

## 证据

共享 artifacts：`/hdfs/share-data-1/users/hjz/Workspace/artifacts/dualstream-layout-comm-20261122`。
`training-summary.json` 保存时延/显存/分阶段结果；`training/*/effective-rank*.json`
记录实际开关、通信dtype和ZeRO配置；逐步loss/梯度范数、所有rank的CUDA event耗时、
退出状态、命令脚本和原始日志均保留。每次重跑应换新目录，保留本次数据。

## 使用与数值验证

optimized YAML 已开启索引复用，视觉 checkpoint 默认仍继承全局设置。
若在本次96 GiB节点上选择显存换速度，可通过配置覆盖：

```yaml
framework:
  qwenvl:
    reuse_vision_attention_layout: true
    vision_gradient_checkpointing: false
trainer:
  gradient_checkpointing: true  # 语言侧仍开启
```

本节点推荐的通信设置（仅对继承环境的进程生效）：

```bash
export NCCL_P2P_LEVEL=SYS
```

下面两项仅用于复现实验候选，**不作为默认推荐**：

```bash
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=8
```

这些 NCCL 调参只在本机/本负载测过，不修改通用启动器默认值。
每卡batch或图像长度改变时，要重新核实显存；正式训练 checkpoint 保存峰值未包含
在这次无保存 benchmark 中。常驻实例保留，实验不会自动启动长期训练。

新增测试包括CPU FP32/CUDA BF16下，24层/4个fusion点的等长与变长输入、
checkpoint开关、输出及所有参数/输入梯度、连续不同输入的索引生命周期和场景隔离；
另外验证视觉checkpoint覆盖不改变语言侧checkpoint与action head FP32。
新测试加framework测试最终16项通过；其余相关回归前面已通过。完整检查覆盖51个
不同用例。测试最初的fixture缺少真实interface会提供的hidden_size适配，修正fixture后
全部通过；历史失败日志保留。首次预取测试因未关闭DeepSpeed发生AcceleratorState
环境冲突，在该测试要求的STARVLA_DISABLE_DEEPSPEED=1下复跑通过。

BF16路径不要求逐位一致，索引路径的单次前反向测试容差为2%，CPU为1e-5。
完整20步baseline对layout组，rank0最大loss相对差约0.035%，grad norm约3.81%；
相同种子并不保证不同kernel layout/通信归约顺序的训练轨迹逐位一致。
这些是短程数值检查，不证明长期收敛或最终任务成功率等价。
