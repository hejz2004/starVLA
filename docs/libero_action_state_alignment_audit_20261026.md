# LIBERO真实数据：原生动作、控制目标与下一观测状态对照

日期：2026-10-26；主机ppu-dev；只读取已有数据与控制目标sidecar，统计在starvla-pai-check内完成，没有新模型推理、仿真rollout或训练。

## 结论

原生action[t]与state[t+1]不是等价的标签。将控制输入换算成物理控制目标后，该目标与下一步实测状态仍有明显差异；不仅夹爪，手臂位置也存在显著差异。不能将此结论说成数据集损坏或已发现一帧错位：控制目标与受动力学/接触影响的响应不同，本来就允许发生。

## 数据与对齐核查

- 原始读取：HJZ的LEROBOT_LIBERO_DATA，1693个parquet，273465帧，原生action为7维，observation.state为8维。
- 辅助目标：WYT minfix labels_precise，只作为由原命令重建的控制目标，不冒充原始数据直接记录的控制器变量。为容器只读分析复制到HJZ报告目录，原WYT文件未修改。
- 1693份parquet及1693份sidecar SHA256均与源manifest匹配；全部原生动作值逐元素匹配；全部anchor位置与实际state位置最大差为0。
- 各文件原始retained_indices用于剔除删除帧间隔；同时排除每条示范第一对记录。最终269999对原始帧相邻的状态—动作参与单步统计，避免将间隔帧强作一个控制步。
- 位置控制目标还独立验证了公式 goal_xyz=state_xyz+0.05*clip(native_action_xyz,-1,1)，所有帧与sidecar目标最大差为0。

## 全量数值

以下位置误差为三维欧氏距离；旋转误差为相对旋转角；夹爪为总开度绝对差。均比较可比物理量，没有直接把7维动作向量与8维观测相减。

| 比较量 | 均值 | 中位数 | 95分位 |
|---|---:|---:|---:|
| 控制器位置目标 vs 下一步实测位置 | 24.34mm | 25.50mm | 41.05mm |
| 控制器姿态目标 vs 下一步实测姿态 | 1.88° | 1.44° | 5.23° |
| 执行器目标开度 vs 下一步实测开度 | 14.78mm | 4.82mm | 54.90mm |

位置目标与下一实测位置的差异，99.22%超过1mm，94.71%超过5mm，不能解释为单纯浮点舍入。

## 可直接定位到文件和帧的例子

Ketchup：libero_object_no_noops_1.0.0_lerobot/data/chunk-000/episode_000001.parquet，第95行到96行，源原始帧索引同样为95到96。

- 原生y控制输入：0.78750002；对应控制目标偏移：39.375mm。
- 真实下一观测的y位移：10.283mm。
- 如果把这个实测位移除以0.05反算OSC输入，只得到0.20566270，约为原命令的26.1%。因此当前未来实测位姿反算方法不能一般性地恢复专家原生控制命令。

| 真实数据样本 | 存储开闭编码 | 还原原生命令 | 重建目标开度 | 下一步实际开度 |
|---|---:|---:|---:|---:|
| ketchup，episode1，行95→96 | 0（闭合） | +1 | 0.00mm | 33.65mm |
| milk，episode7，行110→111 | 0（闭合） | +1 | 0.00mm | 54.10mm |
| tomato_sauce，episode14，行88→89 | 0（闭合） | +1 | 0.00mm | 63.62mm |

这些是具体示范帧，不是物体固定尺寸。持续闭合时，执行器目标可以为0，实际手指受物体接触约束而停在非零宽度。

## 对当前模型的含义

当前UMI16标签是从未来实测state编码而来，预测精确也不等于预测了专家控制目标。将实测未来位姿反算为OSC输入，会改变控制幅值；将实测宽度阈值化或按width_error解码，又会引入意图/执行器内部状态问题。WYT minfix把监督改为控制器目标位姿和执行器目标开度，正是在改变这层监督语义。

上述差异已得到真实数据支持；它在昨晚各任务失败中具体贡献多少，仍需固定权重/初态/采样种子的闭环消融，不能由本次离线统计单独推出。

## 产物

/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Reports/libero_action_state_alignment_20261026/action_vs_next_state_audit.json

/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Reports/libero_action_state_alignment_20261026/ketchup_translation_example.json

完整样本文件路径、行号及xyz数值保存在audit JSON中。
