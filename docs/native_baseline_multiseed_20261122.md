# 原生 QwenPI 两个新训练 seed：交接记录

用户明确要求新增两个随机训练 seed，完整训练原生 baseline 并评测 LIBERO-Plus
成功率；不是给同一个 checkpoint 换评测 seed，也不是训练速度 benchmark。
原 seed42 的代码、训练和评测记录不覆盖。本轮分支为 `experiment/dualstream-bottleneck`，
工作树为 `/hdfs/share-data-1/users/hjz/Workspace/starVLA-dualstream`。

## 任务与配置

| 训练 seed | 动作头初始化 seed | 当前训练任务 | 资源组 |
| --- | --- | --- | --- |
| 788606 | 788607 | tj-dc18aobhq4us73eg9pl0 | pro6000-research |
| 248658 | 248659 | tj-dc18d53hq4us73eg9qkg | pro6000-posttrain |

`pro6000-new` 已更名为 `pro6000-posttrain`，历史与当前资源组 ID 均为
`ng-da6m69agbivc73dic4gg`。用户明确授权 new 组后，seed248658 尚未启动的 research
任务 `tj-dc18aq3hq4us73eg9ptg` 被停止，转为上述独立新任务；旧记录完整保留。
任务仅选择资源组，由平台自动放置，未指定物理节点。

用户要求停止原四卡开发机并以训练任务提交。开发机
`di-dc17as3hq4us73eg9p7g` 已停止，核实状态 Terminated，未删除共享 artifacts。

- 原生 `framework.name: QwenPI`，`dual_stream: null`、`cross_view_attention: null`。
  运行时检查必须加载 Transformers 原生 Qwen3-VL，不含 register/shared/fusion 参数。
- 从官方 Qwen3-VL-2B 权重初始化，动作头独立随机初始化为 FP32，全参数训练。
  不加载 DROID checkpoint，不恢复旧 optimizer。
- 4 GPU，每卡 batch64，累积1，global256，15000更新，warmup750。
  数据/action/归一化/学习率沿用 DualStream optimized LIBERO 配方。
- worker 预处理和 CUDA 预取均开启；全局 gradient checkpointing 开启。
  DualStream 专属索引复用、视觉 checkpoint 覆盖及视觉 attention 分桶未用于原生路径。
- 启动器设置 `NCCL_P2P_LEVEL=SYS`，清除实验用 `NCCL_PROTO` 和
  `NCCL_MIN_NCHANNELS`。实际拓扑、P2P 能力和 transport 日志保存在每组 nodes/node0。
- SwanLab 0.10.5 在线记录；启动器先验证 SDK 版本。训练最初16步自动审计输入、
  架构、各 rank 的 loss/grad norm、global batch 与显存，并输出 STARTUP_ACCEPTED.json。
- 每2000步及最后15000步保存；完整训练状态保留最后一份，评测权重另行 hardlink
  固定，避免 retention 删除后评测无法读取。

## 本次公共预处理改动

worker 预处理与 CUDA 预取在原理上与架构无关，之前仅是实现限定 DualStream：
collator 硬编码了 DualStream 输入接口，并通过 framework 名称白名单拦截原生入口。
本次按实际 framework 选择输入接口，支持 Qwen3-VL 的 QwenPI/QwenFM 和 DualStream；
原生 `QWen3.py` 增加 `return_cpu=True` 路径，worker 无需构造模型或访问 CUDA。
原生默认调用仍把结果放到模型设备，原有语义不变。其他 backbone 尚未在此功能验证。

源码：`starVLA/dataloader/qwen_preprocess.py`、`starVLA/model/modules/vlm/QWen3.py`。
测试：`tests/test_dualstream_optimizations.py` 中 processor/preprocessed/prefetch 用例，
共9项通过，覆盖原生两个入口与DualStream、真实spawn worker、继承分布式环境、
预处理token/图像张量/动作/state/mask逐元素一致、GPU预取以及loss/梯度一致性。
这些修改已冻结到新任务源码，后续工作树改动不影响正在运行的任务。

## 当前评测约定（2026-11-23）

用户最新要求：**每个模型只用评测 seed=7，且只评测最终 checkpoint。**
后训练完成后，先核验完成标记、进程正常退出及权重有限性，再评测最终15000步。
不得自动评测中间checkpoint，也不得自动追加评测seed2026/2027。训练seed与保存策略不变。
该要求同样适用于DROID预训练模型的LIBERO后训练。

固定使用历史协议 `sha256-v1 episode identity + chunk`，`request_seeded=true`：

- 仿真seed与policy启动seed均为7。
- 每次动作chunk的请求seed由 `[7, suite, task_id, trial, chunk]` 派生：
  使用 `eval_protocol.inference_seed`，对紧凑JSON做SHA256，将前8个十六进制字符转为整数。
- 每次推理前显式重置到该请求seed；缺失请求seed直接报错。
  **不是每个chunk都重置成整数7**：不同chunk有不同但可重复的采样seed。
- 不得回退到 `wyt-reference-unseeded-policy`，也不得混用两种协议的结果。
  改协议必须新建评测目录，不能把旧协议的部分回合续跑进来。
- 每次全量LIBERO-Plus四个suite共10030个variant，4GPU/80CPU，保留现有输入、
  动作接口和视频协议。每个模型仅一次seed7评测，不计算跨评测seed的标准差。

Plus policy、controller、worker、protocol已恢复旧seed42评测的相同采样实现；
普通LIBERO parallel evaluator原本即使用该协议。默认入口与要求还记录在
`examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/README.md`。

## 当前最终checkpoint重评任务

权威重评目录：
`/hdfs/share-data-1/users/hjz/Workspace/evaluation-runs/final-plus-seeded-protocol-20261123T024541Z`

| 模型 | 评测seed | 任务ID |
| --- | --- | --- |
| 原生baseline，训练seed788606 | 7 | tj-dc1qjp3hq4us73ega570 |
| 原生baseline，训练seed248658 | 7 | tj-dc1qjqrhq4us73ega57g |
| DualStream v1.1 B32 gated，VLM初始化 | 7 | tj-dc1qjsjhq4us73ega580 |
| DualStream DROID6044 → LIBERO15000 | 7 | tj-dc1qjujhq4us73ega590 |

每个模型继承原有冻结推理代码，只恢复上述采样协议文件。最终权重使用硬链接，
核验训练完成、有限性检查记录、checkpoint大小和相同inode。任务在research/posttrain
两组调度；模型未重新训练，正在运行的任务不受后续工作树改动影响。

- `plan.json`、`submissions.json`：仅4个有效seed7任务与提交回执。
- `validation.json`：4项评测的任务列表、输入/动作处理及采样协议均与历史seed42评测一致。
- `STATUS.json`、`monitor.py`、`monitor-process.json`、`monitor.log`：每分钟读取进度，
  检查实际policy元数据，避免只根据配置认定协议正确。
- `comparison.json`：某模型完整结束后才写入该模型结果，单次评测不报告跨评测seed标准差。
- 每个任务的 `EVALUATION_COMPLETE.json`：最终完成证据，不能把部分回合成功率作为最终成绩。
- `policy-change/seed7-only-20261123T025407Z/`：此前三评测seed计划及8个重复任务的取消记录；
  全部8个已核验为Terminated。不得从归档重新提交seed2026/2027。

## 历史训练与评测记录

原训练交接目录：
`/hdfs/share-data-1/users/hjz/Workspace/artifacts/native-baseline-two-seeds-20261122`

- 两个新训练seed都已完成15000步。首轮最终评测采用unseeded policy，结果为
  seed788606：7414/10030（73.9182%）；seed248658：7389/10030（73.6690%）。
  保留这些结果供审计，当前比较应使用上面的seed7重评；`SEEDED_REEVALUATION.json`指向新目录。
- 旧训练seed42的70.12±0.24来自同一个checkpoint的三个评测seed（7、2026、2027），
  ±为样本标准差，不是三个训练seed的差异；单独seed7为7051/10030（70.2991%）。
- 用户2026-11-22取消中间checkpoint评测：7个未完成任务已核实Terminated，
  6个未提交任务移除。已完成与部分产物保留，不计入最终结果；见`policy-changes/`。
- DROID原unseeded任务`tj-dc1ie4rhq4us73ega0u0`已请求停止，原监督进程停止，
  8908回合部分产物保留；取消与替代记录见新目录的`policy-change/`，不能用于新协议续跑。

每个训练root的 `configs/train.yaml` 是实际配方，`code/` 为冻结源码，
`platform/code-manifest.json` 记录源码hash，`platform/submission.json` 为提交回执；
`outputs/<run_id>/steps.jsonl` 与 `swanlab.json` 是训练记录与曲线入口。
准备目录早期 `native_seed*.yaml` 是关闭worker预处理的草稿，已被正式任务配置取代。
