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

## 自动评测与结果读取

监督程序按现有 Workspace 约定，在每个保存完成的 checkpoint 上触发全量
LIBERO-Plus：2000/4000/6000/8000/10000/12000/14000/15000，共每seed8次。
最终15000评测还要求训练完成标记、权重有限性检查与进程正常退出。
每个评测任务4 GPU/80 CPU，四个suite共10030个variant，沿用现有动作接口和视频协议。
两个训练seed使用相同评测环境seed2026；沿用当前reference的随机policy采样协议
`wyt-reference-unseeded-policy`，不把 policy 采样的波动误解为纯训练 seed 效应。
评测提交到用户已授权的 `pro6000-posttrain`，同样自动调度。

权威入口目录：
`/hdfs/share-data-1/users/hjz/Workspace/artifacts/native-baseline-two-seeds-20261122`

- `run-plan.json`：当前有效任务和评测路径；不要使用移动前的plan或停止的research任务。
- `STATUS.json`：每30秒更新的训练进度、评测状态和错误；平台详情约每120秒刷新。
- `supervisor.py`、`supervisor-process.json`、`supervisor.log`：后台监督脚本、PID和日志；
  使用文件锁防止重复监督，提交前按名称核对平台任务，避免重复申请。
- `final-comparison.json`：仅在两个最终checkpoint都完成10030评测后生成，包含各seed
  的总成功率/分suite结果、均值和样本标准差。最终结果应使用15000步，不挑最好checkpoint。
- `preprocess-tests.log`、`supervisor-tests.json`：预处理验证及保存触发/最终完成门槛/
  hardlink幂等检查。每组平台提交dry-run已通过。

每个训练root的 `configs/train.yaml` 为实际配方，`code/` 为冻结源码，
`platform/code-manifest.json` 记录文件hash，`platform/submission.json` 为提交回执，
`outputs/<run_id>/steps.jsonl`、`swanlab.json` 为实时训练与曲线入口。
`evaluations/step_XXXXXX_libero-plus/EVALUATION_COMPLETE.json` 才是评测完成证据。
准备目录早期 `native_seed*.yaml` 是关闭worker预处理的草稿，已被正式任务配置取代。
本交接文件记录的是任务已提交阶段，不代表15000步训练或成功率评测已经完成。
