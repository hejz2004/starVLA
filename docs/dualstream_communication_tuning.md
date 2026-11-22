# DualStream 四卡通信诊断（2026-11-22）

在 research 常驻环境上复现了每卡 batch 不变时，双卡约 2.52 秒/步、
四卡约 5.57 秒/步的退化。仅设置 `NCCL_P2P_LEVEL=SYS` 后，四卡降至
2.69 秒/步，吞吐从 45.97 增至 95.27 samples/s（2.07 倍）。

## 使用方式

对本次验证过的单机四卡环境，在启动 torchrun/训练进程之前设置：

```bash
export NCCL_P2P_LEVEL=SYS
# 然后执行原训练启动命令，保留原模型、batch、学习率和 ZeRO-2 配置。
```

该设置只作用于继承它的进程，需要新建训练进程才能生效。此次只运行了
有界 benchmark，没有重启或修改现有正式训练任务，也没有修改全局 NCCL 配置。
换节点时需重新核实 GPU P2P 能力、实际 NCCL transport 和短跑性能，不能把本次
单机结果直接推广到其他 GPU、节点拓扑或多机训练。

NVIDIA 对 SYS 的定义是允许跨 NUMA/SMP 互连使用 GPU P2P；它不改变物理拓扑，
也不提供 NVLink：[NCCL 2.27 文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2275/user-guide/docs/env.html#nccl-p2p-level)。

## 环境与根因

- 工作树：`starVLA-dualstream`，`experiment/dualstream-bottleneck`，源码提交
  `dce08d46ba37423dbfdc89eba950906225967dbd`。
- 开发机：`di-dc17as3hq4us73eg9p7g`；平台自动分配至 `pro6000-21`。
- 4 × RTX PRO 6000 Blackwell，48 CPU、384 GB RAM、64 GiB `/dev/shm`。
- PyTorch 2.8.0+cu128、实际加载的 NCCL 2.27.3、DeepSpeed 0.17.6。
- 当前可见 GPU 的 PCI 地址分别为 76:00.0、79:00.0、86:00.0、89:00.0。
  0↔1 和 2↔3 为 PXB，跨这两组为 SYS；全部 GPU 对都报告 P2P 可用。
- 四卡默认日志中，组内为 `P2P/CUMEM`，跨组为 `SHM/direct/direct`，即主机共享
  内存中转；设置 SYS 后跨组也使用 `P2P/CUMEM`。
- 同一机器的跨组双卡默认测试已采用 P2P。因此，不能仅凭“SYS 距离”断言通信
  必然慢；NCCL 对实际参与卡组的路径选择是这里的关键因素。
- DeepSpeed 原本使用 BF16 通信，ZeRO-2、`overlap_comm=false`，两个 bucket
  默认都是 500,000,000 元素。此次无需修改通信精度。

## 真实训练对照

使用现有 `starvla_dualstream_libero_optimized.yaml`：完整 DualStream、真实 LIBERO
数据、每卡 batch64、梯度累积1、GC 开启、BF16、ZeRO-2，既有视觉批处理、worker
预处理和 CUDA 预取全部启用。各组从相同 VLM 权重和随机种子启动；双卡 global128，
四卡 global256，是固定每卡负载的扩展测试，不是相同 global batch 的收敛对照。

每组完成16次 optimizer update，去掉前5次，统计剩余11次中位数。包含数据获取、
前反向和 optimizer update；不包含初始化、checkpoint 写入与日志写入。
相邻日志时间间隔也给出一致结果。测量使用同一训练入口，通过临时 benchmark
包装器记录 CUDA event，在完整 step 结束后同步，未在前向/反向中间插入同步。
SwanLab SDK 为0.10.5，本次计时关闭上传，不导出模型和训练 checkpoint。

| 配置 | 秒/步 | samples/s | 峰值 allocated GiB（各卡最大值） |
| --- | ---: | ---: | ---: |
| 双卡0/1，默认 | 2.517 | 50.86 | 61.92 |
| 四卡，默认 | 5.568 | 45.97 | 51.06 |
| 四卡，仅 `NCCL_P2P_LEVEL=SYS` | 2.687 | 95.27 | 51.06 |
| 四卡，SYS + overlap + 50M buckets | 2.703 | 94.72 | 51.40 |

最后一组同时设置 `overlap_comm=true`、`reduce_bucket_size=50000000`、
`allgather_bucket_size=50000000`，engine-effective 配置已核实。这一组合没有观察到
额外收益，不作为本次推荐配置；此结果不排除其他 bucket 或 workload 上的收益。

rank0 的分阶段 CUDA event 中位数（各列独立统计，不要求严格相加）：

| 配置 | forward | backward（含梯度通信） | engine step（含参数同步） |
| --- | ---: | ---: | ---: |
| 双卡默认 | 0.691s | 1.539s | 0.281s |
| 四卡默认 | 0.697s | 2.286s | 2.567s |
| 四卡SYS | 0.690s | 1.648s | 0.338s |

四卡默认数据等待中位数仅0.82ms。前向计算基本不变，性能改善主要发生在反向和
engine step。仅改变 NCCL 传输策略就消除了大部分退化，支持通信瓶颈的判断；
分阶段时间不是纯通信时间，不能把整个 backward/step 都算成 NCCL 开销。

## 集合通信与数值检查

BF16、1 GiB buffer，每个操作8次，前3次预热；对各 rank 同一轮耗时取最大后
求中位数，并校验输出值。这里只展示 AllReduce：

| 卡组/配置 | 耗时 | 计算得到的 bus bandwidth |
| --- | ---: | ---: |
| 双卡0/1，默认 | 24.8ms | 43.30 GB/s |
| 双卡0/2，默认 | 28.4ms | 37.80 GB/s |
| 四卡，默认 | 123.4ms | 13.05 GB/s |
| 四卡，SYS | 38.3ms | 42.02 GB/s |

bus bandwidth 是按集合通信流量系数折算的指标，不是单条物理链路带宽。
同样测试了4/64/256 MiB，以及 AllGather、ReduceScatter，完整输出保留在 artifacts。
四组训练均正常结束，loss/grad norm 有限。SYS 与四卡默认各 rank 首步 loss 一致，
16步内各卡最大的 loss 相对差异约0.025%。这不是梯度逐位一致性或长期收敛保证。

证据目录：`/hdfs/share-data-1/users/hjz/Workspace/artifacts/dualstream-communication-20261122`。
其中 `training-summary.json`、`training/*/effective-rank*.json`、
`training/*/phases-rank*.jsonl`、各组日志与 benchmark 脚本可用于复核/复现。
重新运行时使用新目录和 run_id，保留本次原始记录。
