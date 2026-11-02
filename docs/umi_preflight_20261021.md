# UMI 正式训练提交前检查（2026-10-21）

## 结论与范围

已检查远程仓库 `/mnt/workspace/Native_Policy/user/hjz/starVLA` 的正式启动脚本与 YAML，并在 `starvla-pai-check` 容器内完成模拟启动和配置测试。
修复启动入口的问题后，当前检查未发现阻止按 4 节点、每节点 16 卡配置启动的脚本或 YAML 错误。
本次未提交训练任务，未进行真实 64 卡通信和显存验证。

检查文件：

- `examples/umi/train_files/run_umi_pretrain_v2.sh`（已修复）
- `examples/umi/train_files/run_umi_pretrain_resume.sh`（语法检查）
- `examples/umi/train_files/starvla_dualstream_umi_v2.yaml`（未改动）
- 训练端 CLI 解析、DeepSpeed 初始化及 WYT 数据 profile/stage 解析代码。

## 发现与修复

1. **裸 KEY=VALUE 被忽略。** 原脚本直接转发，但实际 `normalize_dotlist_args` 只接受 `--key value` 或 `--key=value`。例如 `framework.action_model.use_state=false` 原来解析为空；现在启动器转换后再转发，并验证最终 state=False。
2. **尾部 run 参数可以绕过启动器对输出目录的检查。** 当前明确要求 `RUN_ID`、`RUN_ROOT_DIR`、`CONFIG_YAML` 使用环境变量，resume 使用专门脚本，拒绝尾部覆盖对应字段。
3. **可能继承 smoke 的后端开关。** 原脚本会原样继承 `STARVLA_DISABLE_DEEPSPEED=1` 或 CPU 强制开关；现在立即报错，要求清除这些环境变量。ZeRO 阶段的环境默认值明确为 2；显式外部 DeepSpeed 配置仍可能影响引擎，以启动时 ds-audit 为准。
4. **`.env` 中普通赋值不一定导出给 Python。** 现在读取有效 API key 后显式 export。用临时目录和假 key 验证了子进程能收到 key；没有读取或输出真实 key，也没有发起 W&B 网络连接。

变更只涉及原有 shell；未新增 YAML，未改变训练数学或数据设置。

## 当前正式配置

| 项目 | 配置 |
|---|---|
| 框架 | DualStreamQwenPI |
| VLM | Qwen3-VL-2B-Instruct，加载预训练权重 |
| 双流视觉 | 交互层 [5,11,17,23]，32 shared tokens |
| 归一化 | 条件进入 cross-attention K/V 前和 action decoder 输入前的无仿射参数 LayerNorm |
| 数据 | SHA 固定的 WYT clean20/sameprompt profile，5 个 stages |
| 输入 | 当前三视角图像 + 指令 + 当前绝对 state |
| 动作 | current_eef_relative，双臂 16 维 |
| 时间窗口 | anchor stride=6（5Hz），未来连续16帧（30Hz，约0.533秒） |
| 拓扑 | 4 节点 × 16 卡 = 64 卡 |
| 每卡 batch / 累积 / 全局 batch | 32 / 1 / 2048 |
| diffusion repeat | 1 |
| 精度 / ZeRO | bf16 / 默认 ZeRO-2，无 offload（当前容器默认） |
| 梯度检查点 | False |
| 训练总步数 | 77732 |
| warmup | 2500（当前实际文件值，未改回之前 smoke 的5000） |
| 学习率 | VLM 1e-5；action_model 1e-4；其余 2.5e-5 |
| Scheduler | cosine_with_min_lr，min_lr=1e-6 |
| 优化器 | AdamW，betas=(0.9,0.95)，eps=1e-8 |
| 实际 weight decay | trainer.optimizer.weight_decay=1e-8；外层 trainer.weight_decay=0 当前未用于构造优化器 |
| 梯度裁剪 | 1.0 |
| 保存 | 每2000步，全训练状态 + 平铺模型权重，保留最近3个周期 checkpoint，另存 final_model |
| eval | 间隔10000000，大于本次总步数，本轮不会触发周期 eval |
| 日志 | 每10步；有 W&B key 则尝试启用，否则自动禁用 |
| workers | 每卡2，每节点32，全任务128 |

原 YAML 默认 run_id：`starvla_umi_wyt_relative_state_gb2048_r1`，检查时对应目录尚不存在。建议提交时指定所有节点一致的新 RUN_ID。

## 窗口与 epoch 预算

| stage | 窗口 | optimizer steps（global batch=2048） | 丢弃尾部 |
|---|---:|---:|---:|
| stage_01 | 31842655 | 15548 | 351 |
| stage_02 | 31838251 | 15546 | 43 |
| stage_03 | 31842015 | 15547 | 1759 |
| stage_04 | 31836896 | 15545 | 736 |
| stage_05 | 31838754 | 15546 | 546 |

合计 159198571 窗口，77732 步，每 stage 分别 drop 尾部，共 3435 窗口；不能按合并后的池直接取整为77733。

## 做过的验证

- 三个 UMI shell 的 `bash -n`：通过。
- 模拟四个节点，以临时 Python 替身拦截 torchrun：均生成 `--nnodes=4 --nproc-per-node=16` 及对应 node rank；未启动真实训练。
- `KEY=VALUE` 与 `--key value` 两种 state 消融覆盖经实际解析后均为 False：通过。
- 非法 node rank、已有 run、尾部覆盖 run_id、继承禁用 DeepSpeed/强制 CPU 环境：均在启动前被拒绝。
- WYT profile 固定计划 SHA 校验、五 stage ranges.npy 的实际窗口计算：通过。
- 当前容器内 source/video/quaternion signs、归一化文件与 stage 索引路径存在性：通过。预训练模型配置与权重文件存在（1 个权重文件）。没有逐个扫描全部视频或校验所有权重字节。
- `tests/test_umi_diffusion_repeat.py`：通过。调用真实框架 forward，使用假的 VLM 特征与捕获 action head；输入 batch=2 时 repeat=1/2/3 对应 action head batch=2/4/6，非法 repeat 被拒绝。此项没有加载 VLM 或进行完整模型训练。
- 当前容器 `DeepSpeedPlugin().deepspeed_config`：默认 stage=2，optimizer/parameter offload=none；训练代码显式声明 bf16。

## 提交时的条件

1. 每个节点启动一次脚本。显式设 `NUM_NODES=4 PROCESSES_PER_NODE=16`，`NODE_RANK` 分别为0/1/2/3；`MASTER_ADDR`、`MASTER_PORT` 和 `RUN_ID` 四节点相同。脚本将环境 `WORLD_SIZE` 作为节点数的回退，不能把表示总进程数64的 WORLD_SIZE 误当成节点数。
2. 四节点都应挂载当前代码目录、`umi_wyt_assets`、原始 UMI 数据和 VLM 权重；本次只验证 ppu-dev 容器可见性。
3. 远程 Git status 显示多项 UMI 新代码尚未跟踪。若任务直接挂载当前共享目录可读取；若任务仅从 Git 提交拉代码，必须先将所需代码纳入提交。本次没有替用户进行 Git commit。
4. 双卡96GiB、repeat=1、每卡32的 smoke 只完成1步，第2步 OOM。64卡 ZeRO-2 的模型状态分片更小，但尚未实测64卡峰值，不能保证显存一定足够。未擅自开启 gradient checkpointing。
5. 正常默认启动的引擎审计应包含 `zero_stage=2 micro_batch=32 grad_accum=1 train_batch=2048` 和 bf16 enabled；若不同，应先核对平台注入的 DeepSpeed 环境变量或配置文件。

## 证据

- `playground/umi_preflight_20261021/launch_audit_before.json`
- `playground/umi_preflight_20261021/launch_audit.json`
- `playground/umi_preflight_20261021/resolved_config.json`
- `playground/umi_preflight_20261021/repeat/repeat_budget.json`
- `playground/umi_preflight_20261021/wandb_export_check.json`
- `playground/umi_preflight_20261021/launcher_fix.diff`

当前 shell SHA256：`b471ddfbf6670dad8125876d713a91591a8e22765d41eeba59187e7be0bb207e`

当前 YAML SHA256：`7a1a6b8da6d02486869931bf33bfbe57fea6798ea6ce807369b62e4dcdfce9dd`
