# 昨晚15k模型：逐任务成功率与低分原因分析

分析对象：2026-10-24完成的guarded_width_error评测，原UMI110k预训练＋LIBERO15k检查点；没有重新训练。Object/Long/Goal各500条。

## 结论与范围

总成功率79.0%（1185/1500）。低分集中在Goal的接触操作、开抽屉和阶段顺序，以及Long的多阶段完成度。Object已到96%，不支持模型普遍不会抓取的判断。旧接口的宽物体误开问题大幅缓解，但剩余失败包含抓取定位、姿态/接触执行、放置释放和阶段切换，不能继续全归因于夹爪。

Goal失败160次，其中中抽屉32、上抽屉放碗49、推盘子50、奶油奶酪入碗22，合计153次，占95.6%。Object仅失败20次；Long失败135次。

本次读取1500条已有动作轨迹并按任务/成败统计，另对11个重点任务选取22条失败、10条成功轨迹，共32条视频的五个时刻画面进行人工视觉检查。并未逐帧观看或逐条定因315条失败。稀疏画面能定位明显失败阶段，不能精确证明首次滑落时刻、接触力或所有失败原因。没有启动新rollout。

## 逐任务成功率

### libero_object

| ID | 任务 | 历史threshold | 本次guarded | 成功/次数 |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0% | 88% | 44/50 |
| 1 | pick up the cream cheese and place it in the basket | 2% | 100% | 50/50 |
| 2 | pick up the salad dressing and place it in the basket | 94% | 100% | 50/50 |
| 3 | pick up the bbq sauce and place it in the basket | 92% | 96% | 48/50 |
| 4 | pick up the ketchup and place it in the basket | 100% | 96% | 48/50 |
| 5 | pick up the tomato sauce and place it in the basket | 0% | 96% | 48/50 |
| 6 | pick up the butter and place it in the basket | 44% | 90% | 45/50 |
| 7 | pick up the milk and place it in the basket | 0% | 96% | 48/50 |
| 8 | pick up the chocolate pudding and place it in the basket | 0% | 98% | 49/50 |
| 9 | pick up the orange juice and place it in the basket | 0% | 100% | 50/50 |

### libero_goal

| ID | 任务 | 历史threshold | 本次guarded | 成功/次数 |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 34% | 36% | 18/50 |
| 1 | put the bowl on the stove | 100% | 100% | 50/50 |
| 2 | put the wine bottle on top of the cabinet | 100% | 98% | 49/50 |
| 3 | open the top drawer and put the bowl inside | 2% | 2% | 1/50 |
| 4 | put the bowl on top of the cabinet | 90% | 98% | 49/50 |
| 5 | push the plate to the front of the stove | 10% | 0% | 0/50 |
| 6 | put the cream cheese in the bowl | 2% | 56% | 28/50 |
| 7 | turn on the stove | 96% | 100% | 50/50 |
| 8 | put the bowl on the plate | 98% | 100% | 50/50 |
| 9 | put the wine bottle on the rack | 90% | 90% | 45/50 |

### libero_10

| ID | 任务 | 历史threshold | 本次guarded | 成功/次数 |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0% | 70% | 35/50 |
| 1 | put both the cream cheese box and the butter in the basket | 0% | 80% | 40/50 |
| 2 | turn on the stove and put the moka pot on it | 64% | 54% | 27/50 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 84% | 76% | 38/50 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 86% | 86% | 43/50 |
| 5 | pick up the book and place it in the back compartment of the caddy | 100% | 98% | 49/50 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0% | 66% | 33/50 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0% | 92% | 46/50 |
| 8 | put both moka pots on the stove | 34% | 38% | 19/50 |
| 9 | put the yellow and white mug in the microwave and close it | 80% | 70% | 35/50 |

## 有直接样本支持的失败类型

1. **开抽屉与阶段顺序**：Goal上抽屉放碗仍1/50。失败trial0中，抽屉未打开便开始搬碗；唯一成功trial14先打开抽屉再转移碗。中抽屉失败trial0/23在抽屉附近反复接近却未充分拉开，成功trial2可拉开。支持检查阶段选择与接触位姿，而非简单增加夹爪闭合。

2. **推盘子的接触执行**：Goal推盘子0/50，旧接口也仅5/50。抽样trial0/25未完成持续推移。全部失败轨迹平均25.1%步骤发张开、54.9%保持、20.0%闭合；源示范此前已核查没有稳态闭合命令。因此仍存在实测宽度转意图的不一致，但不能由这些比例单独认定0%就是夹爪导致。需直接检查手指接触位置、推力方向和位姿跟踪。

3. **Long后半程未完成**：开灶放壶失败trial1/20中，炉子已开启，随后壶的抓取/放置失败；杯入微波炉trial0/15中杯子倾倒，未正确放入。碗入抽屉trial2/19有放置在边缘、未完成关屉的现象。两只壶任务的成功轨迹396～504步，失败都到520步上限，时间余量较小；不能据此保证增加上限即可成功。

4. **释放与接触的容错仍不足**：Object字母汤罐失败trial17/35已经抓起并到篮筐处，但未顺利完成放开与撤手。预测仍在约63mm持物宽度附近波动，轨迹出现短暂张开后又闭合；并非完全没有张开命令。此例支持“宽度误差不唯一表达释放意图”，但也需考虑篮筐边缘/手指的接触干涉。

5. **Ketchup两条失败不同**：trial41在初始抓取阶段将瓶子弄倒，随后夹空；58～87步持续闭合，约69/74步的画面已见抓取异常，因此不能把这条失败简单归因于该时段发送张开。trial42已抓起并送到篮筐处，但最后未触发成功，末段有充分张开/保持输出，不是一直被闭合保护锁住；最终物体位置和篮筐接触需进一步确认。

## 可能的共性机制：尚待消融验证

- **未来实测位姿不等于控制器目标**。当前模型监督未来实际EEF位姿，执行端再反算OSC输入；接触、推拉、卡边时，这与专家发送的控制目标/力效果可能不一致。低分集中在抽屉/推盘子与此相符，但单凭视频还不能证明其贡献大小。
- **一次执行16步，接触后更新图像有延迟**。接触或阶段判断出错后，仍需执行预测chunk中的后续目标，可能扩大偏差；随后反复尝试又偏离专家轨迹。缩短执行窗口或使用匹配的native_absolute位姿执行是可隔离验证的方向，本次没有执行此消融。
- **旧权重的宽度标签没有显式释放意图**。guarded只是保护旧闭合区，不能保证任何物体的抓取、释放和阶段切换都正确，也不处理物体被推倒后的恢复。
- 没有证据证明全部低分来自训练步数不足、骨干模型或全局旋转饱和。本次重点任务的旋转命令饱和率为0，许多低分任务平移饱和也接近0；这排除了把问题简单解释为大量输入顶到限幅，但不排除未饱和状态下的跟踪/接触误差。

## 对“退化”的解释边界

旧threshold结果属于历史评测，本轮采样种子流和部分Python/NumPy/JIT环境不同，不是严格同噪声成对A/B。每任务50次，2个百分点仅等于1条episode；Ketchup100%→96%、Long部分任务下降8～10个百分点不能未经成对回放就全部归因于guarded。Goal的0%/2%/36%与Long双壶38%则是当前明确的低成功率，需要优先定位。

## 建议排查顺序（未启动）

先检查Goal上抽屉放碗的阶段顺序和中抽屉/推盘子的接触位姿，再检查Long壶/杯的抓取姿态及放置完成条件；最后针对Object剩余少量失败研究释放去抖和恢复。若验证执行接口，固定权重、初态和每chunk采样种子，将手臂执行方式与夹爪规则分开消融。不要直接更换所有阈值或同时更改多个控制环节。

## 保存位置

/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Reports/libero_failure_analysis_20261024

文件：trace_statistics.json、selected_episodes.json、selected_command_runs.json、release_and_grasp_examples.json、visual_observations.json、contact_sheet_00.jpg至contact_sheet_07.jpg。

原始1500条结果及完整审计：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Evaluation/libero_object_goal_long_guarded_parallel_20261024

本报告：docs/libero_per_task_failure_analysis_20261024.md。

## 补充：已定位的内部目标记忆与释放短脉冲问题

读取本次实际robosuite源码确认：20Hz控制、0.002秒物理步，每个env.step调用25次gripper.format_action；Panda speed=.01，对称两指目标总开度每物理tick约变0.4mm，即每个持续开/闭的控制步累计约10mm，直到物理端点。输入0保留内部目标。这个目标与接触约束下的实际宽度不同。

根据初始10步张开及保存的离散命令，可离线重建内部目标（未直接记录的控制器变量，故标为推算值）：

- 字母汤trial17：170步开始发张开时，实测62.99mm，内部目标推算仍为0mm；170～175共6步张开，内部目标到60mm，仍小于物体撑出的约63mm。176步预测从76.17mm跳回61.88mm，解码器随即改闭合，目标从60回50mm。trial35也有释放被下一chunk打断的现象，但数值不同：169～175连续7步张开，推算目标短暂到70mm；176步实际宽度仅63.82mm，随即改闭合，目标回到60mm。它还体现了实际关节响应相对目标的滞后。
- 因而这些样本并非“有张开命令就已充分松开”：张开先要消除此前积累的闭合目标偏置，短释放段被下一chunk的闭合打断后，仍可能夹住物体。此机制由实际执行代码和已记录命令支持；接触/摩擦仍可能共同影响实际释放。
- Ketchup trial42不同：168～175连续8步张开，推算内部目标已到80mm，之后实测也接近78mm。它的失败不适合解释为内部目标一直未打开，应优先检查放置位置和篮筐边缘接触。

这也说明只比较预测宽度与当前实测宽度的解码器遗漏了执行器内部状态；若以后改进，应同时考虑意图状态、内部目标以及释放持续时间。不能直接把误差反馈换成内部目标误差就保证正确，因为旧权重预测的仍是实测宽度。

数据见inferred_actuator_target_examples.json。以上为离线重建，没有重新执行策略或仿真。

分析于2026-10-26接续完成；本报告日期后缀对应2026-10-24的评测产物。确认未启动新评测或修改模型、控制器。
