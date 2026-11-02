# LIBERO-plus anchored-goal 并行评测

2026-10-29：默认采用最新 `libero-anchored-osc-goal-binary-v1`；旧 UMI16 宽度接口已退出执行路径。

新模型重复评测的完整步骤见 [PAI DLC LIBERO-plus 操作记录](../../../../../docs/pai_dlc_libero_plus_runbook.md)。新任务使用 `launch.template.json`；本目录的 `launch.json` 保留最后一次历史续跑配置。

## 运行配置

- 单节点 16 PPU、160 CPU；每卡一份 BF16 模型、20 个常驻 CPU 仿真 worker，共 320 worker。
- 每组绑定 10 个实际允许的 CPU，20 个 worker 轮流绑定这 10 核，即每核 2 个 worker；模型预处理仍共享末尾两核，模型最多两个 Torch 线程。CPU 总申请保持 160 核。
- 全局内存队列按剩余最大步数加权分发；优先保留套件局部性，空闲 worker 可以领取其他套件。模型不随套件切换重载。不使用 DDP，不合并推理 batch。
- 任务为 `(suite, variant_id, trial=0)`，四套共 10030 条、七类扰动。正常任务失败计入成绩；环境/进程异常直接停止，保留已完成结果，修复后通过 --resume 继续。
- 指令协议为 `libero-plus-bddl-v1`：沿用环境规则，定位 BDDL 时去掉 `_view_` 及后续运行时后缀；普通扰动读取该 BDDL 的 `language_instruction`，语言扰动保留文件名中的 `_language_` 变体标识并读取对应改写内容。只整理空白，不从文件名生成指令。两路 RGB 图像各旋转 180° 后，按主摄、腕摄顺序用 PIL BICUBIC 缩放到 224×224；策略输入只有 `image`、`lang`。
- 直接复用普通 LIBERO `eval_libero.py` 的观测处理和动作执行：双摄顺序、224 缩放、16 步固定锚点、实时 controller reference、OSC7、二值夹爪都一致。策略 RNG 与 WYT 一样不显式设定或逐请求重置；仿真 seed=7。
- OSMesa、BLAS、OpenMP、OpenCV、ImageMagick 和视频编码限制线程；Numba JIT 保持关闭。

## 入口

`launch.json` 复用 `dlc-submit-01/job-ppu-posttrain.json` 的镜像、NAS 挂载、VPC、驱动和资源：单节点 16 PPU、160 CPU、1600Gi 内存。其 UserCommand 调用 `launch.sh`，依次安装系统依赖、`requirements.txt`、editable starVLA，再启动评测。`requirements.txt` 引用已有后训练 `requirements-ppu.txt` 和仿真依赖，并固定 NumPy 1.26 兼容的 OpenCV / h5py。PPU Torch、Transformers 等镜像包通过安装时的版本约束保留。

`launch.sh` 默认使用 `/opt/ac2/bin/python`，直接安装到镜像环境，不依赖 DSW 预装的 simulation overlay。Plus 源码默认使用仓库 `playground/Artifacts/libero_plus_shared/source`；`run_dlc.sh` 也是完整 bootstrap 入口。

默认复用 WYT 的完整公共资产 `/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets`，评测仅读取公共资产；DLC 挂载与之前任务一致，整个 lpy NAS 挂到 `/mnt/workspace`（RW），public NAS 挂到 `/mnt/nas/public`（RO）。公共资产已经包含标准物体，不默认混入其他资产目录。

环境已准备时，直接使用仓库的 `scripts/run_libero_plus_parallel.sh`：

```bash
STARVLA_PYTHON=/opt/ac2/bin/python \
LIBERO_PYTHON=/opt/ac2/bin/python \
LIBERO_HOME=/shared/LIBERO-plus \
LIBERO_ASSETS=/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets \
LIBERO_SITE_PACKAGES=/shared/libero_plus_sim_runtime \
CKPT_PATH=/shared/run/checkpoints/steps_15000_pytorch_model.pt \
NUM_CARDS=16 CPU_BUDGET=160 WORKERS_PER_CARD=20 \
RUN_ROOT=/shared/Evaluation/new_plus_run \
  bash scripts/run_libero_plus_parallel.sh
```

默认 checkpoint 为本仓库 `multiview-umi110k-libero-posttrain` 的 15000 步模型；旧宽度 checkpoint 会在加载模型前被拒绝。共享 Plus 源码、BDDL、初态及完整资产必须可见。原共享源文件不会被评测代码修改，运行目录复制一份源码，BDDL、初态及资产采用链接。

两卡冒烟：设置 `EVAL_MODE=smoke NUM_CARDS=2 CPU_BUDGET=20`，保持每卡 20 worker；默认四套各选择前5个变体，共20个回合。`--task-ids`、`--suites` 可进一步缩小验证范围。`--dry-run` 仅输出计划；允许规划超过本机可用核数的 CPU_BUDGET，但真实启动会拒绝超额 CPU 或不可见设备。

## 输出与恢复

- `protocol.json`：检查点、训练配置、评测代码、Plus Python 源码、分类表及运行环境身份。
- `launch.json`：本次设备、worker 数、CPU 绑核与 PID。
- `${RUN_ROOT}.bootstrap.log`：安装依赖与控制器标准输出；`policies/device_*/policy.log`、`workers/*/run.log`：模型和仿真持久日志。
- `summary.json`、`<Suite>/summary.json`：每约 10 秒更新总体、套件和扰动类别进度。
- `<Suite>/results/*.json`：以变体身份命名的逐回合记录。
- `<Suite>/videos/<episode_key>/<perturbation>_suc.mp4` 或 `_fail.mp4`：用任务独立目录避免重名；文件名不含语言指令。主摄在左、腕摄在右的 512×256 画面上方增加信息栏，显示套件/任务编号、扰动类别和实际 BDDL 指令（自动换行），10 fps。信息栏只用于视频，不进入策略观测。
- `<Suite>/traces/*.npz`：实际 OSC7 动作及控制轨迹。
- `queue_state.json`、`workers/*/progress.json`、`errors/`：任务分配、步数心跳与异常记录。
- `completion.json`：全部任务及必要产物完成后才写入。

`--resume` 从完整结果恢复，允许改变设备、worker 数和 CPU 分配；模型、任务、动作协议、代码或仿真版本变化会拒绝混用。已经正常失败的任务也会被保留，不会为了提高成功率而重跑。视频与轨迹先提交，结果记录最后原子提交；相同语言描述的不同变体不会覆盖视频。输出目录有进程锁。

旧的固定范围 `run_card.sh` 已明确退役；`run_pai_node.sh` / `run_gates_and_full.sh` 是新入口别名。旧 flat-format 汇总工具仅供历史记录，不参与新运行。

## 验证

在 `starvla-pai-check` 容器中执行 `python -m unittest discover -s .../pai_parallel/launchers -p 'test_*.py'`，并提供已固定的仿真 PYTHONPATH、LIBERO_CONFIG_PATH、OSMesa 环境。测试覆盖目标/夹爪接口、完整变体清单、320-worker 分配、错误处理、断点恢复、资源链接以及真实 ffmpeg 视频解码。真实双卡验证与 16 卡计划的证据保存在远程仓库 `playground/Reports/libero_plus_goal_binary_migration_20261029/`。

2026-10-30 DLC bootstrap：同一原始镜像的全新 Docker 中完成依赖安装、editable starVLA 安装与双卡 × 10 worker 的四套 20 回合验证，退出码 0。证据和提交回执见 `playground/Reports/libero_plus_dlc_bootstrap_20261030/validation_and_submission.json`。测试回合的任务成功数为 1/20；该样本用于验证安装与执行链路，不是完整测评成绩。

2026-10-30 长指令视频修复：新增视频按扰动类别及 suc/fail 命名并加顶部字幕；320 worker 共享 160 CPU。此前完成的回合保留原始轨迹和视频画面，视频仅改短名称与任务目录。续跑前将旧协议及结果 JSON 归档，并记录仅涉及视频产物和 worker 默认数的协议迁移；不放宽一般性的 checkpoint / 输入 / 动作协议一致性检查。验证、迁移与续跑回执见 `playground/Reports/libero_plus_video_w20_20261030/`。

依赖一致性按实际安装版本记录；不对整份 dist-info/METADATA 文本做哈希，因为 Gym 0.25.2 的构建会随机重排 Requires-Dist。仿真实际版本、模型与输入/动作代码仍需匹配。

2026-10-31 起结果 JSON 记录 `task_name`、`instruction` 和 `instruction_source_bddl`。历史 2026-10-30 成绩使用原始 `task.language`；新指令协议使用新运行目录，沿用现有协议比较机制。
