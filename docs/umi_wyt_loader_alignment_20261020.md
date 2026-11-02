# UMI dataloader 与 WYT 对齐实施报告

> 当前正式配置（2026-10-21）：diffusion repeat=1，64卡×每卡32×累积1=global2048，一轮77,732步。详见第10节。

日期：2026-10-20。修改直接落在远程 `ppu-dev:/mnt/workspace/Native_Policy/user/hjz/starVLA`。测试均在 `starvla-pai-check` 容器内运行。

## 1. 完成内容

原有 `examples/umi/train_files/starvla_dualstream_umi_v2.yaml` 已切换到 `umi_indexed` 后端，没有新增 YAML。接入 WYT 冻结版本的真实数据读取模块、数据索引、四元数符号映射、归一化统计量，以及五个训练 stage。默认 `framework.action_model.use_state: true`，可覆盖为 false。

对齐的来源是此前已根据 Oct20 实际启动入口核实的冻结包：

```
/mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/
umi_clean20_recovery_meanval_4x16_19000_20261015_v1/frozen_bundle.zip
```

此前核实的包 SHA256：`f692cb1c1e3898011034d4e961eab75024bf776d32136ce375146c6387e4a211`。

本次从该包直接提取以下八个模块，未改写其内部读取规则，容器中已逐个核对导入内容的 SHA256：

- `starVLA/dataloader/umi_indexed_dataset.py`
- `starVLA/dataloader/umi_language.py`
- `starVLA/dataloader/umi_normalization.py`
- `starVLA/dataloader/umi_pose.py`
- `starVLA/dataloader/umi_quaternion.py`
- `starVLA/dataloader/umi_sampler.py`
- `starVLA/dataloader/umi_video_batch.py`
- `starVLA/dataloader/umi_video_time.py`

## 2. 当前样本定义

| 项目 | 实施后的规则 |
|---|---|
| 图像 | 当前时刻 head_left、wrist_left、wrist_right，顺序固定，RGB，224×224 |
| 样本起点 | 每个有效区间内，从区间起点每隔 6 帧取一个 anchor |
| 动作标签 | 30Hz 原始数据的未来连续 16 帧，即 t+1 到 t+16，约 0.533 秒 |
| 语言 | Luna `annotation.instruction`，完整 17 行窗口遵守 same_segment 协议 |
| 过滤 | 使用相同的离线有效区间、训练/验证划分及 20 个 episode 排除列表 |
| 四元数 | 使用 episode 连续符号映射；不再对每个未来目标单独 canonicalize |
| 动作坐标系 | 每只手分别转换为当前末端坐标系下的相对位置、相对旋转；夹爪保留未来值 |
| state | 当前时刻双手的世界绝对位姿与夹爪，共 16 维，形状 [1,16] |
| 归一化 | 配套固定 mean/std；使用其极小 std 处理规则，不裁剪到 [-1,1] |
| 视频匹配 | WYT 的 PTS 边界、1ms engineering slack 和首 anchor 容差规则 |
| 失败行为 | 抛出带样本索引的异常，不随机换另一条样本 |
| 采样 | block_size=512，interleave_blocks=1，seed=42，由 Accelerate 分片一次 |

旧配置预测 t+6 到 t+96，跨度约 3.2 秒；这次已经改为上述连续未来帧定义。旧的 stride6 q01/q99 统计文件不再用于这条预训练路径。

## 3. state 消融

唯一开关仍为：

```yaml
framework:
  action_model:
    state_dim: 16
    use_state: true
```

启动时可覆盖：

```bash
--framework.action_model.use_state false
```

两组都读取当前状态，并用它构造相同的相对动作标签，也返回相同的归一化 state。True 时构建 state encoder 并输入 state token；False 时不构建 state encoder、不输入 state token。图片、语言、动作标签、划分、统计量和 sampler 不因开关改变。

若进行严格的 state 对照实验，应对齐共有模块的初始权重。现有 state encoder 的创建会消耗随机数，仅设置相同 seed 不保证两组所有共有模块初始权重一致。本次没有改变 action head 的初始化策略。

## 4. 五个 stage

按照 WYT 的五份 episode 分组依次训练，模型、优化器和 scheduler 连续继承。每个阶段只丢弃不足一个全局优化器 batch 的尾部，完成预算后进入下一个阶段；不会重新初始化模型。

| Stage | anchor stride=6 的窗口数 | global batch 1536 的更新数 | global batch 2048 的更新数 |
|---|---:|---:|---:|
| stage_01 | 31,842,655 | 20,730 | 15,548 |
| stage_02 | 31,838,251 | 20,728 | 15,546 |
| stage_03 | 31,842,015 | 20,730 | 15,547 |
| stage_04 | 31,836,896 | 20,727 | 15,545 |
| stage_05 | 31,838,754 | 20,728 | 15,546 |
| 合计 | 159,198,571 | 103,643 | 77,732 |

更新数 = floor(stage 窗口数 / 全局优化器 batch)。全局优化器 batch = 每卡 microbatch × rank 数 × 梯度累积。

保留现有每卡 batch=24 和累积=1，所以 64 卡时自动得到 103,643 步。WYT 每卡 32、64 卡时对应 77,732 步，已验证预算计算结果一致。

原 YAML 的 `trainer.max_train_steps: null` 表示自动训练上述完整一遍。设置为正整数可截短为 smoke test，不能超过整轮预算。预算在模型和 scheduler 构建前解析，最终数值写入运行配置。

checkpoint 记录当前 stage、epoch（当前一遍方案为 0）、已提交的全局样本游标、当前阶段更新数及数据身份。只在成功处理训练 microbatch 后推进游标，只在优化器边界保存 checkpoint，不把预取当成已经训练。

恢复时直接定位 sampler 游标，不逐 batch 解码快进。切换 stage 会关闭旧 iterator/workers/cache，仅保留一个活跃 stage 的 loader。

## 5. 工程接入与数据资产

新适配模块：

- `starVLA/dataloader/umi_wyt.py`：对接现有 factory、导出真实 mean/std 参数、解析阶段预算与恢复身份。
- `starVLA/training/trainer_utils/umi_stage_data.py`：stage 生命周期、提交游标、恢复和独立验证 batch。

修改现有文件：

- `starVLA/dataloader/__init__.py`：注册 `umi_indexed`。
- `starVLA/training/train_starvla.py`：接入 stage/checkpoint；spawn worker 不初始化分布式 Accelerator；在优化器边界保存、记录完整数据配置。
- `starVLA/model/tools.py`：mean/std 动作反归一化分支，不裁剪、不把四元数维度误当夹爪二值化。
- `deployment/model_server/policy_norm_processor.py`：识别新 UMI checkpoint 自带的固定统计量。
- 原 UMI YAML 与两个已有正式/恢复启动脚本：接入新后端，支持 CLI 覆盖与明确恢复，保护旧 run。

运行目录中的 `dataset_statistics.json` 包含动作和状态的有效 FP32 mean/scale 及来源信息。stage 访问记录位于 `data_access/<stage_name>/dataset_access.json`。

训练索引、验证索引、相对动作统计量与使用约定已复制到容器可见目录：

```
/mnt/workspace/Native_Policy/user/hjz/umi_wyt_assets/
clean20_sameprompt_relative_20261015_v1
```

保留了 WYT 原文件；只复制索引资产，不复制原始 Parquet 或视频。重复 SQLite 只复制一份，在自己的新资产目录内部使用硬链接共享。五个训练 stage 的 11 个独立二进制 payload（5 个 ranges、5 个 cumulative、1 个共享 SQLite）已按原 meta 的 SHA256 校验。

原始标注数据、视频和符号映射从 NAS 只读访问。索引内源路径通过显式 source_root/video_root 映射到当前容器路径。

## 6. 测试与结果

测试入口：`tests/test_umi_wyt_alignment.py`。

| 测试 | 结果 |
|---|---|
| 8 个 WYT 模块的导入内容 SHA256 | 全部匹配提取时记录 |
| 五个训练 stage 索引二进制 SHA256 | 11 个独立 payload 全部匹配 |
| stage_01 的 32 个真实低维窗口 | 连续未来帧、排除规则、语言窗口、状态/动作形状检查通过 |
| 相对动作独立核对 | 用 SciPy Rotation 独立计算位置和四元数，与反归一化后的标签一致，绝对容差 2e-6 |
| 图像读取 | 三视角顺序、224×224、真实视频解码通过 |
| 单进程 stage/恢复 | 阶段内恢复、阶段边界恢复、错误数据身份拒绝恢复通过 |
| 双 PPU、梯度累积=2 的数据分片 | 相同检查通过，样本索引符合 rank 分片，未重复分片 |
| state=True/False 的真实小型 DiT | 每组完成 2 次优化；True 的 state encoder 有非零梯度；False 没有 state encoder 参数 |
| mean/std 推理反归一化 | FrameworkTools 与 PolicyNormProcessor 均与训练 normalizer 一致；输入 3.25 也不裁剪或二值化 |
| 实际 VLATrainer/Accelerate checkpoint | 模型、AdamW、scheduler、RNG 和游标恢复成功；重放下一步得到与连续训练逐位相同的权重 |
| Python 编译及启动器 shell 语法 | 通过 |

完整模型额外验证：

- 原 UMI YAML，继承 Qwen VLM 权重，DualStreamQwenPI，state=True。
- 使用空闲 PPU 1，microbatch=1、梯度累积=2，共 2 个优化器步骤。
- 使用真实 UMI 数据、配置中的两个 spawn workers，禁用 W&B。
- 两步日志 action_dit_loss：**5.0332317352、4.2093725204**。
- 对应记录的梯度范数：15.0124、11.8623；样本游标为 2、4。
- 训练正常结束，生成独立 smoke run；没有覆盖旧 run。

测试证据：

```
playground/umi_wyt_checks_20261020/data/data.json
playground/umi_wyt_checks_20261020/stage/stage.json
playground/umi_wyt_checks_20261020/distributed/distributed.json
playground/umi_wyt_checks_20261020/model/model.json
playground/umi_wyt_checks_20261020/checkpoint_final/checkpoint.json
playground/umi_wyt_checks_20261020/full_model_smoke.log
```

完整模型 smoke run：`playground/Checkpoints/smoke_umi_wyt_stage_state_20261020_v1`。

## 7. 使用方式与范围

正式启动仍使用原脚本，提供新的 RUN_ID：

```bash
RUN_ID=starvla_umi_wyt_state_on_v1 \
bash examples/umi/train_files/run_umi_pretrain_v2.sh
```

PAI 多机拓扑环境变量沿用原脚本约定；本例没有在本次任务中启动正式长训。

state=False 另开独立 run：

```bash
RUN_ID=starvla_umi_wyt_state_off_v1 \
bash examples/umi/train_files/run_umi_pretrain_v2.sh \
  --framework.action_model.use_state false
```

恢复同一新版 run 时，用原有 resume 脚本并显式给出 RUN_ID。恢复脚本默认读取该 run 的 config.full.yaml，自动保留其 state、microbatch 和梯度累积，无需重复填写 False 等覆盖；数据、batch 或 state 身份不一致会拒绝 strict resume。

旧 loader 的 checkpoint 可用于新 run 的权重初始化，但不能把旧的数据游标当成新五阶段流程的位置。新训练和恢复入口会在写入配置前检查已有 run，避免覆盖旧实验。

本次对齐数据读取与 stage 组织。现有 DualStream 模型结构、每卡 batch=24、优化器/学习率方案没有改为 WYT 全套模型训练配置。当时模型的 diffusion repeat 仍为2；此项现已按第10节改为可配置，并在 UMI YAML 中设为1。

独立验证使用固定 held-out monitor 的前 16 个窗口，不消耗训练 sampler，也不改变模型 RNG；这不是全量验证集评测。

当前证据覆盖单卡完整模型短跑、双卡数据分片和 CPU 小模型完整 checkpoint 恢复；尚未进行 64 卡 DeepSpeed 长时间训练。两步有限 loss 说明流程可运行，不证明长期收敛。mean/std 空间的 loss 也不能直接与旧 q01/q99 空间的数值比较。


## 8. 预训练 YAML 与启动脚本同步更新

根据后续要求，更新了两份已有 YAML 和三个已有启动脚本，没有新增 YAML：

- `starvla_dualstream_umi_v2.yaml`：正式配置，五个 stage、预训练 VLM、state=True、microbatch=24、累积=1；`max_train_steps: null` 按实际全局 batch 计算完整一遍预算。显式保存完整训练状态。
- `starvla_dualstream_umi_smoke.yaml`：与正式配置使用相同数据规则和模型设置；10 个优化步骤、microbatch=1、累积=16，单卡有效 batch=16。没有周期 checkpoint，但训练器仍保存 final_model。
- `run_umi_pretrain_v2.sh`：统一启动逻辑，读取 YAML 默认 run 名称和输出目录，支持环境变量和原有 CLI 覆盖；从任意工作目录调用均先定位远程仓库，补全 PYTHONPATH。已有 run 会拒绝重复创建。
- `run_umi_pretrain_resume.sh`：默认加载指定 run 的 config.full.yaml 并显式启用 strict resume，不复制覆盖原配置。
- `run_umi_smoke.sh`：复用正式启动器，默认单节点单卡、关闭 DeepSpeed 和 W&B，自动生成含时间与进程号的唯一 run 名称。NUM_PROCESSES 可调整本节点卡数。

在训练容器内运行 shell 语法检查、YAML/stage 预算检查，并拦截最终 torchrun 调用验证参数；验证过程没有启动真实训练。

单卡 smoke（在训练容器内，可从任意目录执行脚本绝对路径）：

```bash
CUDA_VISIBLE_DEVICES=1 bash examples/umi/train_files/run_umi_smoke.sh
```

state=False：

```bash
CUDA_VISIBLE_DEVICES=1 bash examples/umi/train_files/run_umi_smoke.sh \
  --framework.action_model.use_state false
```

每条上述相对路径命令的工作目录应为仓库根目录。脚本内部会统一切回仓库根目录。NUM_PROCESSES=2 时，如保持累积=16，有效 batch 为 32。


## 9. 简洁 YAML 与相对 EEF 约束（2026-10-21）

用户提出 vla_data 参数过多后，改为“固定版本的数据 profile + 两个资产根目录 + 常调参数”。两份原有 YAML 的 vla_data 都从 62 行缩到 9 行，没有新建 YAML。

正式配置现在为：

```yaml
  vla_data:
    # Fixed WYT rules, five stages and statistics resolve from its pinned JSON plan.
    dataset_py: umi_indexed
    profile: wyt_umi_relative_v1
    assets_root: /mnt/workspace/Native_Policy/user/hjz/umi_wyt_assets/clean20_sameprompt_relative_20261015_v1
    umi_root: /mnt/nas/public/roban_umi
    action_pose_repr: current_eef_relative  # Enforced by profile; world is rejected.
    per_device_batch_size: 24
    num_workers: 2
```

smoke 使用同一 profile，per_device_batch_size=1；其 10 steps、累积=16、默认无 W&B 的设置保留。use_state 开关仍位于 framework.action_model，不影响数据标签。

实现位于现有 `starVLA/dataloader/umi_wyt.py`：

1. 从 assets_root 读取之前复制的 WYT `wyt_input_plan.json`，按固定 SHA256 校验版本。不是再手工复制一份完整默认参数。
2. 从计划读取细粒度语言、相同片段过滤、相机、归一化、视频容差、缓存、episode 排除列表和五个 stage 等规则。
3. 将原计划的 NAS 源路径映射到 umi_root，将索引和统计量映射到 assets_root。stage 更新预算仍按实际 batch/world size/累积重新计算。
4. YAML/CLI 可覆盖高级调优参数；当前预训练入口只接受 `current_eef_relative`。省略该字段时 profile 自动补齐；传入 world 或其他动作表示会报错。
5. 构建 dataset 后再次核对 action_pose_repr，保证接入的是实际执行当前末端坐标变换的 wrapper。
6. 展开在模型/调度器构建和运行配置保存前完成；config.full.yaml/config.yaml 仍保留完整有效规则、stage 和 profile 来源 SHA。

profile 绑定计划 SHA256：`31a47613300b9c2426f3b93386c41796f6d0aca3e49beeedbc0c164441c29901`。未知 profile、被修改的 JSON 计划、显式 world 均拒绝。WYT 八个原始读取模块未修改。

为保持旧 checkpoint 兼容，profile 名称和两个根目录等配置选择元数据不进入旧版数据身份计算；实际展开后的路径、规则、统计量和样本顺序继续参与校验。旧的完整展开配置仍可使用。

新增测试 `tests/test_umi_wyt_profile.py`，在 starvla-pai-check 容器中验证：

- 精简前后有效配置逐项一致，64 卡 global1536 的总预算仍为 103643。
- 精简前后数据身份完全相同，支持旧版已展开配置。
- 8 个真实窗口的原始/归一化 state、action 和语言完全一致；当前三视角图像逐位一致。
- state=False 时得到相同标签。
- 省略 action_pose_repr 仍使用相对 EEF；直接构建 loader 或规划 stages 时传入 world 均报错。
- 错误 profile 和修改后的计划 JSON 被拒绝。
- 五个 stage 和完整展开配置可保存；smoke 有效 batch 仍为 16。
- 实际 VLATrainer/Accelerate 小模型 checkpoint 恢复后，模型、AdamW、scheduler、RNG 和样本游标正确，重放下一步权重逐位一致。

证据：

```
playground/umi_profile_checks_20261021/profile/profile.json
playground/umi_profile_checks_20261021/profile/resolved_config.txt
playground/umi_profile_checks_20261021/checkpoint/checkpoint.json
```

这次改动验证了配置等价和恢复链路，没有重新启动完整 VLM 训练或正式预训练。


## 10. diffusion repeat 与 global batch 2048（2026-10-21）

根据用户要求，更新正式预训练配置：

```yaml
framework:
  action_model:
    repeated_diffusion_steps: 1
datasets:
  vla_data:
    per_device_batch_size: 32
trainer:
  gradient_accumulation_steps: 1
  max_train_steps: 77732
```

这是现有 YAML 内的配置节选，没有新增 YAML 或增加冗余的 global_batch_size 输入字段。实际全局数据 batch 在运行时由 64×32×1 计算为2048，写入展开配置的 stage_global_batch_size。diffusion repeat 表示每条数据内部的 flow-matching 噪声/时间采样次数，不计入数据窗口 batch 或 epoch 样本数量。

正式默认 run_id 改为 `starvla_umi_wyt_relative_state_gb2048_r1`，避免与之前 batch24/repeat2 的实验混用。Smoke 仍为 microbatch1、累积16、10步，也显式配置 repeat=1。

修改 `DualStreamQwenPI.py` 和 `QwenPI.py`：去除 forward 中强行覆盖 repeat=2 的语句，实际读取 framework.action_model.repeated_diffusion_steps。只接受正整数；0、负数、布尔值、浮点数和字符串报错。未提供该字段的旧配置保持实际默认 repeat=2。

从实际五个 ranges.npy 按 anchor_stride_frames=6 重新计算：

| Stage | 窗口数 | global2048 更新数 | 丢弃的尾部窗口 |
|---|---:|---:|---:|
| stage_01 | 31,842,655 | 15,548 | 351 |
| stage_02 | 31,838,251 | 15,546 | 43 |
| stage_03 | 31,842,015 | 15,547 | 1,759 |
| stage_04 | 31,836,896 | 15,545 | 736 |
| stage_05 | 31,838,754 | 15,546 | 546 |
| 总计 | 159,198,571 | 77,732 | 3,435 |

当前 stage 实现逐阶段丢弃不足一个全局优化器 batch 的尾部，因此一轮是 sum(floor(N_stage/2048))=77,732步。若把全部窗口合成一个池后统一取整，则 floor(159198571/2048)=77,733步；这与当前五阶段执行方式不同。

正式 YAML 将 max_train_steps 从自动值 null 改为明确的77732，并更新相关注释。若以后改变卡数或梯度累积，应设回 null 重新计算完整一遍，或显式指定短训练预算。

新增测试 `tests/test_umi_diffusion_repeat.py`：调用两个真实 framework 的 forward，用轻量特征/动作头替代权重加载，验证 repeat=1、2、3 对动作、state、多层特征、attention mask 和 DualStream action mask 的实际复制。输入batch2时，对应动作头batch2、4、6；省略配置时保持batch4。有限loss与反向传播通过，非法repeat被拒绝。

测试同时核对了正式配置中的repeat1、每卡32、累积1、77732步，以及实际索引的五阶段预算和尾部数量。证据：

```
playground/umi_repeat_gb2048_checks_20261021/repeat_budget.json
```

相关数据/profile 回归检查已按新的每卡32预算更新，比较时对齐基线 batch，继续验证读取规则和相对 EEF 标签不变。本次没有启动正式64卡训练。
