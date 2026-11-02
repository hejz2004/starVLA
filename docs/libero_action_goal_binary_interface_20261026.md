# LIBERO后训练新接口：原生动作重建目标、统一观测锚点、二值夹爪

完成日期：2026-10-26；开发机ppu-dev。用户选择“控制器目标相对chunk起点”，并要求不自动识别检查点语义。代码、标签生成和接口验证已完成；未启动后训练或新模型成功率评测。

## 当前约定

版本：`libero-anchored-osc-goal-binary-v1`。

模型保持H=16、每步16维：

- 0:3：OSC目标位置相对于chunk起点观测的位置，在起点坐标系表达，单位米。
- 3:7：OSC目标姿态相对于chunk起点观测的xyzw四元数。
- 7：原始action的二值开闭意图，0闭合、1张开。
- 8:16：单臂填充0，训练loss屏蔽。

保持双视角224×224图像与语言指令；参考state用于标签构建和执行转换，不作为模型额外输入。

## 数据标签怎样生成

原LeRobot parquet和视频不覆盖、不重新编码。读取同一行的原始7维action和观测state：

1. 手臂标签来源是action，而非下一帧实际到达的位姿。按标准OSC输入边界缩放，将action[i]解释为当前state[i]处发出的命令；位置目标为 `p_goal[i] = p_obs[i] + 0.05 * clip(action[i,:3], -1, 1)`。旋转命令按0.5rad缩放并作用于当前姿态；旋转输入为0时保留原生控制器上一旋转目标。
2. 为保留第一条记录之前的旋转控制历史，使用已经核验的原始控制历史receipt初始化旋转目标；其后由本次代码从action独立重建。源parquet、历史receipt及重建结果均逐文件/逐目标核验。参考receipt的后续目标只用于交叉验证，不用于替代本次运动标签计算；原WYT文件不修改。生成后训练和评测不依赖WYT代码或运行目录。
3. 对图像所在的chunk起点t，仅使用一个固定观测锚点 `(p_obs[t], R_obs[t])`。每个i=t..t+15的目标都转换为 `R_obs[t]^T * (p_goal[i]-p_obs[t])`，姿态为 `q_obs[t]^-1 * q_goal[i]`。不是后一目标减前一目标，也不把原生增量简单累加。
4. 夹爪从原始 `action[i,6]` 读取，经 `>0.5`得到0/1；不使用qpos差、实际持物宽度或执行器目标开度。
5. 图像I[t]对应目标goal[t]..goal[t+15]，与当前动作发出时刻对齐。只采样完整16步窗口，不跨episode、不用尾帧重复填充。

例：当前位置依次为0、5、11mm，三步原生x输入都是0.8，则控制目标为40、45、51mm。相对于第一个观测的标签也是40、45、51mm；不是实测5、11mm，也不是累计命令的40、80、120mm。

## 已生成的数据

根目录：`/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Datasets/LIBERO_OSC_GOAL_BINARY_V1/`。

- 1693条示范、273465帧、248070个完整H16窗口。
- Object：454条，60174个窗口；Goal：428条，45622个窗口；Spatial：432条，46490个窗口；Long：379条，95784个窗口。
- `labels/<dataset>/episode_*.npz`：重建目标、观测锚点、原动作与来源索引。
- `statistics.json`：新标签的归一化统计与接口声明。
- `manifest.json` / `READY.json`：源数据、标签哈希及生成完成凭证。

位姿前7维使用q01/q99仿射归一化，不裁剪；夹爪0/1保持不变，后8维置0。物理几何运算float64，模型训练标签float32；动作loss只使用前8维。服务端使用同一统计和同一逆变换。

## 评测接口

默认明确指定 `ACTION_INTERFACE=anchored_goal_binary`，没有auto选项，不根据检查点元数据自动切换到旧接口。检查点配置/元数据只用于确认所选接口是否匹配和还原训练归一化；不匹配直接报错。

1. 接收模型16维预测并反归一化，用chunk起点观测锚点恢复各步控制目标。
2. 更新当前原生OSC参考位姿，处理EEF body与grip site固定旋转偏置；以目标相对当前控制器位姿的误差反算标准OSC输入，按原生[-1,1]边界执行。目标姿态与上一控制器姿态目标相同的近零差异，保留原生零旋转保持语义。
3. 第8维预测大于0.5输出-1张开，否则+1闭合，包含恰好0.5时闭合；与官方评测的夹爪转换一致。新接口不使用width_error、41.9mm保护线、intent_latch或0保持命令。
4. 最终执行标准 `env.step([ux,uy,uz,urx,ury,urz,g])`，保留原生OSC及默认Panda夹爪，不使用连续开度替代控制器。

这是在用户要求的16维、固定chunk锚点表示下，对齐官方控制目标与开闭命令的语义；不是声称网络输出维度和动作H都改成官方发布示例的7维/H8。

## 训练入口

默认新配置：`examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero_action_goals.yaml`。

`examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh` 已默认使用该配置；run_id默认带goal_binary前缀，并拒绝在已存在的不同标签协议run中恢复。保留原模型与训练超参数，使用原UMI预训练权重初始化；旧state标签配置文件保留供显式选择。

训练配置中的关键字段：

```yaml
data_mix: libero_all_anchored_osc_goal_binary
action_type: libero-anchored-osc-goal-binary-v1
action_horizon: 16
controller_goal_root: /mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Datasets/LIBERO_OSC_GOAL_BINARY_V1
normalization_statistics: /mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Datasets/LIBERO_OSC_GOAL_BINARY_V1/statistics.json
include_state: false
action_mode: abs
state_action_dtype: float32
```

其中action_mode=abs用于关闭通用数据层的二次相减；实际标签的锚点相对转换已经由本接口完成。

## 后续评测命令

在ppu-dev仓库根目录，等新接口后训练产生权重后：

```bash
CKPT=/absolute/path/to/new_run/checkpoints/steps_N_pytorch_model.pt \
OUTPUT_DIR="$PWD/playground/Evaluation/new_goal_binary_eval" \
ACTION_INTERFACE=anchored_goal_binary \
  bash scripts/run_libero_eval_2ppu.sh --suites libero_object libero_10 libero_goal
```

两张PPU并行入口继续保留，CKPT必须显式提供，避免默认误用旧15k权重。宽度标签旧模型对应的 legacy_umi16 评测接口已删除；不能通过切换名称使用这两个新入口。原生7维权重手动选 `ACTION_INTERFACE=native_osc7`。不会自动选择。

旧权重没有被改造为新语义；这次未启动新后训练，不能据此报告新成功率。

## 验证与文件

- 全量1693条原生目标重建：位置最大差0；旋转最大差4.012e-8rad；所有完整窗口归一化往返最大差2.086e-8。
- 四套件真实双视角样本读取、16×16标签和mask、checkpoint统计导出与服务端逆变换已通过。
- 与真实原生OSC控制器的目标生成对照通过，覆盖非零旋转与零旋转历史保持；无需加载模型。
- 固定锚点、非累计增量、夹爪二值及恒等缩放、显式接口不匹配报错测试通过；原生7维路径及旧接口拒绝检查通过。
- Python编译、shell语法和本次新增行空白检查通过。

验证记录：`playground/Reports/libero_action_goal_binary_upgrade_20261026/`。
主要实现：

- `starVLA/dataloader/libero_action_contract.py`
- `examples/simBenchmarks/LIBERO/tools/build_action_goal_binary.py`
- `examples/simBenchmarks/LIBERO/train_files/data_registry/action_goal_binary_config.py`
- `examples/simBenchmarks/LIBERO/eval_files/action_goal_binary.py`
- 配套修改数据注册表、PolicyNormProcessor、ModelClient及标准/并行评测入口。

重建命令的历史输入来自本仓库已保存的原始数据核验快照：

```bash
python examples/simBenchmarks/LIBERO/tools/build_action_goal_binary.py \
  --data-root playground/Datasets/LEROBOT_LIBERO_DATA \
  --output /new/empty/output/directory \
  --history-sidecars playground/Reports/libero_action_state_alignment_20261026/source_sidecars \
  --history-manifest playground/Reports/libero_action_state_alignment_20261026/source_sidecar_manifest.json
```
