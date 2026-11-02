# WYT LIBERO 动作接口对照

日期：2026-10-23；主机ppu-dev。WYT目录只读，报告写在HJZ仓库。

## 核心结论

WYT并非所有LIBERO实验都使用同一动作接口。找到四类监督语义和五种主要执行模式。即使模型头都是16维，第8维也可能分别表示未来实测宽度、原生开闭命令或执行器目标开度。当前可核查的WYT代码没有使用HJZ这条“宽度>41.9mm即张开”的固定中点解码。

用户提出的40k预训练后91.60%尚未在可访问的已保存评测中唯一定位，不能给这个数字指定接口。发现30k/40k/50k nativeosc准备分支，但其runs为空，检查文档明确learned_policy_evaluated=false，因此不能拿准备配置证明91.60%的来源。另有20k预训练后97.8%的连续开度分支，以及96.35%的native7/pad16分支，均已从8个worker的各2000条记录独立重算。

## 接口一览

| 分支/模式 | 模型监督 | 夹爪执行 | 主要差异 |
|---|---|---|---|
| 早期UMI16，delta | t+1..t+16实测EEF位姿与两指间距；相对chunk起点 | 比较预测宽度与当前实测宽度，1.5mm容差，输出-1/0/+1 | 没有41.9mm全局分界；位姿反算delta OSC |
| 早期UMI16，native_absolute | 同上 | 同上 | 绕过位姿反算近似，向原OSC set_goal传set_pos/set_ori；校正EEF body与grip site固定旋转偏置 |
| OSC goals / osc_goals | 原生OSC目标位姿+原生夹爪命令 | 第8维直接clip到[-1,1]后交给默认Panda | 目标是controller goal而非未来实测EEF位姿；夹爪是命令而非宽度 |
| Native OSC7-in-16 / native_osc | 原始7维控制命令；前6维位移/axis-angle，夹爪放index7 | 直接env.step；默认Panda取符号 | identity归一化；index6及index8..15不计loss；H16从t开始 |
| UMI-aligned / minfix | 原生controller目标位姿+执行器目标开度；相对记录图像的anchor | PandaApertureController直接连续跟踪执行器目标开度 | 第8维不是受物体限制的实测指间距；替换format_action，按原生速率移动内部目标 |
| PI05-style native7/pad16 | 原始7维命令连续放index0..6，index7..15补0 | 取回前7维直接env.step | q01/q99仿射不裁剪；padding也参与loss；不是使用PI0.5模型 |

## 早期UMI16：同样的标签，不同的解码

代码位于：/mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_10k_ppu_20261020_v1/code/examples/LIBERO/umi16_control.py

gripper_command逻辑：
```python
width_error = predicted_width - current_measured_width
command = 0 if abs(width_error) <= 0.0015 else -sign(width_error)
```

例如物体把夹爪撑在54mm、模型也预测54mm时，它发0而不是张开。0保持默认Panda内部执行器目标，不代表强制把实际宽度锁到54mm，也不是独立力控制指令。若之前目标已闭合，它不会仅因实测54mm>41.9mm而松开。该规则仍可能受预测噪声、接触与开闭过渡影响，不能保证解决全部问题。

早期两次完整记录：UMI10k为79.20%，其中Object98.2%；UMI20k为79.25%，其中Object98.6%。这已说明“同样用未来实测宽度监督”不等于“同样受到全局阈值误开的影响”。这些模型与HJZ结构/训练不同，不是只改夹爪的严格消融。

## UMI-aligned：最关键的语义变化

标签生成工具：/mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/tools/build_original_video_sidecars.py

代码从原始记录图像对应的observation.state和匹配的native commands构造controller目标；保留被删除noop帧对夹爪内部目标的历史影响。标签为c.goal_pos、转换为body表达的c.goal_ori和aperture.target_width()，不是state[t+1]实测宽度。

因此，物体令实际指间距停在54mm时，闭合命令对应的执行器目标仍可以继续向0mm靠近。模型学习的是这个目标，底层位置执行器向内施力，实测宽度则由接触约束决定。

执行器适配：/mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/umi_aligned_control.py

PandaApertureController把目标宽度映射为[w/2,-w/2]两指位置，再映射到原执行器归一化current_action；每个物理tick以原Panda speed向目标靠近，目标到达后保持。只应用0..80mm物理界限，不用统计分位数中点决定开闭。它临时替换gripper.format_action，绕过默认np.sign。

最终minfix实际调用链：CollectionCalibratedOSCAdapter → NativeDeltaPrecisionAdapter → PrecisionUMIAlignedOSCAdapter。位姿从当前自有机器人controller状态反算受原边界约束的OSC输入；夹爪仍由PandaApertureController执行。它没有在推理时读取专家未来轨迹。代码名step_absolute不代表最终一律绕过原生delta控制，必须看具体继承覆盖。

归一化是q01/q99仿射，不裁剪标签或预测；不是mean/std。执行时才将开度限制到物理范围。

## 已重算的完整评测

| run | 预训练 | 后训练 | 模式 | Spatial | Object | Goal | Long | Overall |
|---|---|---|---|---:|---:|---:|---:|---:|
| libero_umi_minfix_20k_7500updates_8x64_all_eval_v1 | UMI20k | 7500 updates，GB512 | umi_aligned | 98.6% | 99.4% | 97.8% | 95.4% | 97.8%（1956/2000） |
| libero_pi05pad_20k_7500updates_8x64_all_eval_v1 | UMI20k | 7500 updates，GB512 | native_osc | 97.8% | 99.6% | 97.4% | 90.6% | 96.35%（1927/2000） |

本轮分别从8个worker读取2000条episode，唯一(suite,task,trial)数量、成功数与summary一致；每任务50次，没有把smoke和正式结果混合。UMI-aligned六个继承链源码文件SHA256与历史worker协议一致。未重新执行策略、未重新训练、未独立观看全部轨迹。

这组对照还同时改变监督语义、夹爪执行、归一化数值、mask/padding等，不能把1.45个百分点差异全归为某一项。

## 对此前归因的修正

1. “若监督和执行完全一致，应有一致的结构性问题”这一质疑成立；实查显示接口不一致。宽度阈值误开仍是HJZ当前实现的明确风险，但不是LIBERO固有限制，也不是所有UMI16模型必然遇到的问题。
2. 目前证据支持“当前阈值存在语义错配且与Object分化高度一致”，不支持“全部差距已经确定由该阈值造成”。仍缺HJZ同权重、同初态、同噪声的闭环A/B。
3. 不能只复制WYT的连续开度controller给HJZ旧模型：旧模型输出实测宽度，WYT新模型输出执行器目标，含义不同。旧权重可先用早期width-error适配做诊断；长期应对齐监督和执行语义后训练。
4. 高成功率分支的位姿监督也改为controller goal，不只是夹爪从二值改连续；不能把所有收益归给连续夹爪。

## 证据路径

- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_train_libero_plus_pipeline_20261022_v1/runs/libero_umi_minfix_20k_7500updates_8x64_all_eval_v1/evaluations/pipeline_libero_2000_v1/summary.json
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_pi05pad_20261022_v1/runs/libero_pi05pad_20k_7500updates_8x64_all_eval_v1/evaluations/pipeline_libero_2000_v1/summary.json
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_10k_ppu_20261020_v1/runs/libero_umi16_fromclean20_10k_10epoch/evaluations/native_absolute_mujoco323_4suite_50trials_16ppu/summary.json
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_10k_ppu_20261020_v1/runs/libero_umi16_fromclean20_20k_10epoch/evaluations/native_absolute_mujoco323_4suite_50trials_16ppu/summary.json

- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_10k_ppu_20261020_v1/code/examples/LIBERO/umi16_control.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_10k_ppu_20261020_v1/code/examples/LIBERO/train_files/data_registry/data_config.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/umi_aligned_control.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/umi_native_delta_precision.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/umi_collection_calibrated.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/umi_aligned_normalize.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/code/examples/LIBERO/train_files/data_registry/umi_aligned_config.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_minfix_20261022_v1/tools/build_original_video_sidecars.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_30k40k50k_nativeosc_ppu_20261021_v1/code/examples/LIBERO/native_osc16.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_30k40k50k_nativeosc_ppu_20261021_v1/code/examples/LIBERO/osc_goal16.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_30k40k50k_nativeosc_ppu_20261021_v1/code/examples/LIBERO/train_files/data_registry/native_osc_config.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_30k40k50k_nativeosc_ppu_20261021_v1/configs/qwenpi_libero_nativeosc16_40k.yaml
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_pi05pad_20261022_v1/pi05pad_actions.py
- /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments/libero_posttrain_clean20_20k_pi05pad_20261022_v1/pi05pad_eval.py

结构化来源哈希、评测协议与统计见同目录wyt_libero_action_interfaces_20261023.json。WYT文件只读，本轮只新增HJZ仓库报告。
