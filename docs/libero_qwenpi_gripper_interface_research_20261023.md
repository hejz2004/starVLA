# 官方 QwenPI 的 LIBERO 动作接口与连续夹爪离散化调研

日期：2026-10-23；主机：ppu-dev。用户要求暂停控制器改进与评测，先核清官方接口和相关做法。本轮只做源码、配置、文档调查和报告保存，没有启动训练或 rollout，没有修改控制逻辑或默认配置。

## 核心结论

官方 QwenPI 的 LIBERO 路径使用连续数值动作头，但夹爪维监督的是开闭命令编码，而不是未来实测夹爪宽度。连续输出与离散意图并不矛盾：连续模型可以拟合 0/1 标签，执行前再阈值化。

我们现有 UMI16 后训练把未来实测宽度作为夹爪标签；物体接触使实际宽度不能唯一决定开闭意图。把统计归一化区间中点当作类别边界，会把正确的宽物体持握状态译成张开。单纯增加迟滞、保持状态或改阈值，不能恢复标签中已经丢失的命令信息。

## 1. 官方训练—输出—执行链

调查对象为官方发布的 StarVLA/Qwen3-VL-PI-LIBERO-4in1 及 starVLA_dev 公共实现，不将其他 QwenPI 配置一概等同于该检查点。

| 项目 | 官方发布模型 / libero_all 路径 | 当前用户模型 |
|---|---|---|
| 动作维度 | 7：平移控制量3、旋转控制量3、夹爪命令1 | 16；前8为相对位置3、相对四元数4、实测宽度1，后8掩码 |
| 时间长度 | 发布配置 H=8 | H=16 |
| 夹爪监督 | 标准化开闭编码，0闭、1开 | 未来两指实际关节位置之差，单位米 |
| 夹爪归一化 | Libero4in1DataConfig 只对前6维设 min_max；夹爪保持原编码 | 实测宽度跟随 q01/q99 缩放及裁剪 |
| 执行转换 | g>0.5 输出 -1张开，否则 +1闭合 | 旧默认：宽度>约41.932mm张开，否则闭合 |
| 手臂执行 | 直接使用预测的平移、旋转控制量 | 从预测未来实测位姿逆算 OSC 误差，再缩放、裁剪 |

证据链：

- [发布配置](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1/raw/main/config.yaml)：action_dim=7、action_horizon=8、data_mix=libero_all。配置名 delta_qpos 不代表这里输出七个机器人关节角；实际分量以数据注册表和执行代码为准。当前类默认 H=16 不能覆盖此发布配置的 H=8。
- [数据注册表](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/train_files/data_registry/data_config.py)：Libero4in1DataConfig 的 action_keys 为 x/y/z/roll/pitch/yaw/gripper，action_indices=range(8)；libero_all 使用 libero_franka。另一个 openpi_libero_all 使用不同归一化配置，不能混用其结论。
- [QwenPI 模型](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/framework/VLM4A/QwenPI.py)：动作经 flow-matching 头预测为连续数组。夹爪浮点数不是经过专门概率校准的分类概率。
- [环境适配器](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py)：取前三维为 world_vector、随后三维为 rotation_delta、第七维为 open_gripper；服务端返回已反归一化结果。该文件虽保留 sticky 字段，但注释和执行路径表明 LIBERO 未消费这些字段，不能把官方高分归因于 sticky。
- [官方评测](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py)：_binarize_gripper_open 为 1-2*(g>0.5)。因此 g=0 是闭合，不是底层保持指令。

### 为什么它能适配、成功率正常？

接口方面，它直接学习示范中的控制意图，再映射到环境的原生开闭命令。夹住宽物体时，实际宽度即使为54mm或63mm，目标标签仍可保持闭合；不需要从接触后的宽度猜测是否释放。

[发布卡片](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1/blob/main/README.md)报告100k时 Object98.6%、四套件平均97.5%。这些是发布方报告值，本轮没有用原始 rollout 独立复算。官方4B与当前2B、训练步数、动作表示及执行时间长度等也不同，不能证明全部成功率差距由夹爪接口造成。能确认的是：官方链路避开了当前这类宽度到意图的结构性误译。

## 2. 连续量首先要分清是哪一种量

1. 开闭命令的浮点编码，例如0闭1开：可以阈值化，0.5有类别编码意义。
2. 目标开口宽度：有物理单位，要求匹配的位置/力控制器；不是天然的开闭类别。
3. 实测开口宽度：是动作、物体接触和动力学共同作用的结果，不能普遍唯一还原命令。
4. 相对开闭事件或速度：0通常表示不产生新事件，需保留此前的逻辑状态；不能按绝对宽度处理。

例如，实测宽度54mm可能对应正在张开，也可能对应持续闭合但被瓶身挡住。预测宽度比当前宽度略大，也不充分说明策略想释放；可能是噪声、指垫形变、时序错位或未来抓取姿态变化。这是从控制和观测关系得到的分析，不是某个基线的性能结论。

## 3. 相关实现如何处理

### 3.1 OpenVLA / Octo：离线过渡标签与相对事件展开

[OpenVLA data_utils](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/utils/data_utils.py)的 binarize_gripper_actions 对0..1动作信号采用 >0.95开、<0.05闭，并逆向扫描，将中间段赋为后续明确端点。若末尾一直处于中间值，其实现保留末尾值，不能声称输出无例外全为二值。rel2abs_gripper_actions 则以死区检测相对事件，前向传播上一次开闭状态；初始化带有数据假设。Octo有同类[数据工具](https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py)。

对我们适用的经验：有原始命令时优先保留命令；只在信号语义满足假设时重标注。OpenVLA 的 Bridge 重标注甚至将手臂改为实测位姿差，同时保留原夹爪动作，说明手臂与夹爪可以采用不同监督来源。

不能照搬的部分：后续端点是离线可用的未来数据，在线执行时不存在相同信息；预测 chunk 也不是已实现的未来观测。宽物体接触后若一直停在中间宽度，可能永不触及“完全闭合端点”，逆向补标签反而可能把整个持握段标成张开。这一失败情形是依据算法假设的推论。

[OpenVLA执行工具](https://github.com/openvla/openvla/blob/main/experiments/robot/robot_utils.py)还明确区分0..1编码转换和环境符号反转。归一化空间、物理单位和开闭符号必须分别核对，不能重复反转或在错误空间阈值化。

### 3.2 SimplerEnv：在线事件重复 / sticky

[SimplerEnv Octo适配器](https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/policies/octo/octo_model.py)按机器人分别处理夹爪。Google robot 路径由相邻预测开闭量的变化构造事件，并重复执行若干步；WidowX路径有不同规则。

可借鉴的是“保留逻辑意图、过滤短暂波动”，不是复制具体步数。重复时长应换算为秒并匹配控制频率；过度保持也会错过真实释放。连续多个同一chunk预测高度相关，不能当作独立证据。所有状态应在episode重置。

### 3.3 PerAct：显式开闭分类

[PerAct实现](https://github.com/peract/peract/blob/main/agents/peract_bc/qattention_peract_bc_agent.py)用两个夹爪logit、分类损失和argmax表达开闭。它说明可把离散意图与连续几何分开建模，但并不意味着我们必须改网络：仿照官方在现有连续动作头中训练0/1标签也可表达开闭。PerAct不是本次LIBERO对照性能来源。

### 3.4 UMI / Franka：保留连续目标，但匹配控制语义

[UMI WSG控制器](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/wsg_controller.py)接收连续位置时间点并执行位置跟踪，同时记录位置、速度、力。它提供连续执行的参考，并不提供将实测宽度无损还原为二值命令的办法。

[libfranka Gripper](https://frankarobotics.github.io/libfranka/0.16.1/classfranka_1_1Gripper.html)区分 move(width,speed) 与 grasp(width,speed,force,epsilon_inner,epsilon_outer)。抓住物体允许非零宽度并有容差；目标位置、实际位置和夹持力不是同一件事。此为真实机器人接口原则，不是当前MuJoCo环境的可直接替代函数。

## 4. 当前 LIBERO 的“保持”要怎样理解

[robosuite v1.4 PandaGripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)使用 sign(action) 累加内部执行器目标：负号向张开更新，正号向闭合更新，零停止更新这个内部目标。输入可以是浮点数，但幅值在该层被取符号，不能用0.2与0.8精细控制开口。

因此：模型编码 g=0（闭合）与环境输入 u=0（保持内部目标）不是一个概念。零不表示松开，也不保证夹持力足够；若在接触瞬间就停止收紧，内部目标可能没有积累足够闭合偏置。上层逻辑“保持夹持”在此接口下可以继续发闭合命令，但力仍受执行器与物理参数限制。不能将此扩展成所有硬件都应无限持续闭合。

## 5. 转换与训练时需要核对的项目

- 明确标签是命令、目标还是观测；不要由相同数值范围推断语义。
- 标注单位、正负方向、开闭端点；夹爪使用独立归一化mask或明确变换。统计分位数中点不是物理分类边界。
- 核对 action[t] 与 observation[t]、state[t+1]、未来窗口的时间关系及控制延迟。源命令不能不检查就与未来实测位姿按同一索引拼接；应沿源动作时间戳定义每个未来目标对应的命令。
- 把“保持当前开口”与“保持接触夹持”分开；不要把触物停滞或非零宽度自动当作未闭合/要张开。
- 迟滞、事件确认、最短驻留只能抗噪，不能补回未知意图；必须检查其造成的闭合或释放延迟。
- 连续手臂量可以平滑，离散开闭跨阶段平均可能产生没有意图意义的中间值；应单独设计夹爪的融合规则。
- chunk执行长度决定接触后多久才能重新看到图像；状态机计步单位应是实际执行控制步。重置时清空缓存和意图。
- 容错宜结合接触、力、滑移等反馈；位置误差本身不充分表达抓取状态。
- 验收既要测宽物体改善，也要测ketchup闭合、运输、释放三个阶段，保持相同环境和种子；少量选中失败初态的改善不是整体成功率。

## 6. 对当前工程的建议（尚未实施）

首选一个可隔离的后训练对照：保留现有16维结构和手臂相对位姿表示，将夹爪索引7的监督替换为原始LIBERO开闭命令的0/1编码，单独处理归一化；核对时间对齐后重新后训练。推理时按官方符号映射执行。这样不必同时改变骨干、动作头和手臂接口，便于检验夹爪标签语义的影响。可将实际宽度保留为状态或辅助预测，但不要让它唯一决定释放。

这需要新训练与新统计配置，不能只把旧权重的宽度数值按0.5解释。UMI预训练和LIBERO后训练的同一槽位语义会改变，需明确阶段/embodiment及mask，避免混合数据时静默冲突。若长期要跨域保持统一连续语义，则应监督执行器目标开度并配套连续、接触感知的控制器；不能把实测宽度重命名为执行器目标。

既有 guarded_width_error 和 intent_latch 仅是旧权重的补偿性试验，各只完成4条选定ketchup困难初态，均3条成功但失败初态不同，不构成完整回归通过。用户暂停后不再开展候选规则或100条验证。不能承诺某方案保证ketchup不降；在固定回归集验证并通过前不替换默认接口。

## 7. 调研留档与证据边界

报告位置：本文件。源链接、下载状态与SHA256见 playground/Reports/libero_gripper_interface_research_20261023/source_manifest.json。外部资料通过浏览工具阅读；远程直接抓取部分文件因网络失败，manifest保留真实失败状态，不冒充完整源码存档。已成功下载的文件保存在 sources/；另保存 remote HEAD 的官方相关源码，只作本仓库基线快照，不声称与网页动态分支逐字一致。

远程HEAD：4507931a625c844404c4a76128fb116536d8ca7c。公共链接多指向动态分支，结论限定于本日所读版本。旧评测和示范统计沿用已审计产物，见 docs/libero_eval_bimodality_analysis_20261023.md 与 docs/libero_object_width_error_eval_20261023.md。本次没有新增成功率测量。
