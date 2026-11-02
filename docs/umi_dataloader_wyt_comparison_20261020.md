# wyt 与 HJZ 的 UMI 预训练 dataloader 对照调查

日期：2026-10-20。所有调查针对远程文件；没有修改训练实现或训练 YAML。

## 1. 结论

两套代码使用相关的 UMI 原始轨迹，但实际训练样本定义并不相同。最重要的差异依次是：

1. **预测时间跨度不同。**wyt 每 6 帧选一次观察起点，但预测接下来连续 16 个 30 Hz 帧，跨度约 0.533 秒；HJZ 同时对动作帧做 stride=6，预测 t+6 到 t+96，跨度 3.2 秒。
2. **语言监督不同。**wyt 后期读取当前帧的 Luna 细粒度 annotation.instruction，离线限制整个窗口属于同一个标注片段；HJZ 当前读取 meta/episodes 的 tasks，属于 episode 级任务描述，没有同片段约束。
3. **数值目标不同。**wyt 后期采用 episode 连续四元数符号、相对当前 EEF 的动作、mean/std 标准化；HJZ 采用逐目标四元数规范化、相对当前 EEF 的动作、q01/q99 缩放并裁剪。
4. **样本池和使用顺序不同。**wyt 从清洗池剔除 20 个 episode，保留 701 个 episode 为验证集，将训练集分成五个阶段；HJZ 在一个 loader 中使用全部 320,712 个清洗后 episode。
5. 图像主干输入有相同部分：都是当前观察时刻的 head_left、wrist_left、wrist_right 三张 RGB 图，缩放至 224×224；没有在一个样本里提供未来 16 张图片。

因此，wyt 的训练 loss 与 HJZ 的训练 loss **不能仅按“同一 UMI 数据集、同一种模型”直接比较**。加入相同 LayerNorm 并没有使两套训练任务自动一致。

## 2. 调查依据与版本范围

HJZ 仓库：

    /mnt/workspace/Native_Policy/user/hjz/starVLA

wyt 实验根目录：

    /mnt/workspace/Native_Policy/user/wyt/a100/wyt/experiments

主要采用以下冻结材料：

- 2026-10-20 续训入口 umi_resume_10000_77732_20261020_v1/launch_resume_dlc.sh。
- 该入口明确引用的原实验 umi_clean20_recovery_meanval_4x16_19000_20261015_v1/frozen_bundle.zip。
- 可直接查看的 umi_world8_zero2_dlc_20261019_v1/code。已将其中五个关键 loader 文件与上述冻结 ZIP 逐字比较：umi_indexed_dataset.py、umi_pose.py、umi_language.py、umi_normalization.py、umi_sampler.py，内容完全相同。
- 早期对照采用 qwenpi_r1_dit_fp32_zero2_b32a1_64gpu_2000_20261014_v3/frozen_bundle.zip。这条早期实验不能代表 wyt 后期所有数据定义。

“当前 wyt 方案”在本报告中指上述后期 clean20/sameprompt/relative 方案，而不是仅根据文件夹名字猜测版本。未读取凭证内容，未启动 wyt 的训练程序。

## 3. 一条样本是如何构造的

### 3.1 wyt 后期方案

~~~text
离线清洗 / 训练验证划分 / 同语言片段筛选
    ↓
每个 stage 的 ranges.npy + cumulative.npy + metadata.sqlite3
    ↓
每个有效范围内以 6 帧间隔选观察起点 t
    ↓
读取源数据连续 17 行：t, t+1, ..., t+16
    ├─ 当前状态：第 0 行，两只手的世界位姿与夹爪，共 [1,16]
    ├─ 未来动作：后 16 行，共 [16,16]
    ├─ 语言：第 0 行的 annotation.instruction
    └─ 图像：t 时刻三路相机，各 224×224
    ↓
应用离线 episode 连续四元数符号表
    ↓
未来世界位姿转为“相对 t 时刻各自手部坐标系”的动作
    ↓
固定 mean/std 标准化 state 和 action
    ↓
块随机采样 → DataLoader → Accelerate 按 rank 分配 batch
~~~

注意：四元数符号修正实际上发生在原始 state/action 组装之前；图中按逻辑阶段排列，不表示先将错误符号的姿态转成相对动作。

### 3.2 HJZ 当前方案

~~~text
扫描 / 缓存 meta/episodes，应用 kept membership
    ↓
每个 episode 从 0 开始，以 6 帧间隔列举观察起点 t
    ↓
读取连续 97 行，再选 t, t+6, ..., t+96
    ├─ 当前状态：第 0 行，两只手的绝对位姿与夹爪 [1,16]
    ├─ 动作：选出的 16 个未来时刻，转为相对当前手部坐标系
    ├─ 语言：episode 元数据 tasks + task_index
    └─ 图像：t 时刻三路相机，各 224×224
    ↓
对 state / 每个未来相对四元数分别选最大绝对分量为正的符号
    ↓
固定 q01/q99 缩放，全部维度裁剪到 [-1,1]
    ↓
块随机采样 → DataLoader → Accelerate 按 rank 分配 batch
~~~

两者都不是“每个未来动作相对上一个动作”的逐步增量；后期方案中的所有未来动作均以同一个当前观察时刻 t 为参考。

## 4. 时间：同样写着 5 Hz，预测的动作并不相同

| 项目 | wyt 后期方案 | HJZ 当前方案 |
|---|---|---|
| 原始数据 FPS | 30 | 30 |
| 观察起点间隔 | anchor_stride_frames=6 | anchor_stride=6 |
| 动作帧间隔 | 固定连续源帧，stride=1 | frame_stride=6 |
| 动作标签 | t+1, …, t+16 | t+6, t+12, …, t+96 |
| 动作标签频率 | 30 Hz | 5 Hz |
| 最后目标距离当前观察 | 16/30≈0.533 秒 | 96/30=3.2 秒 |
| 起点相位 | 每个有效 range 的开头重新计数 | 每个 episode 从 0 开始 |

直接代码证据：

- wyt UMIIndexedDataset.read_lowdim：stop = anchor + 17；返回 values[:1] 和 values[1:]。
- wyt UMIIndexedDataset._open_arrays：anchor_stride_frames 仅改变 ranges 内的起点数量，不抽稀未来 action 行。
- wyt 索引 meta.json 明确记录 fps=30、action_stride_frames=1。
- HJZ read_window：span = self.stride * self.horizon，读 span+1 行，再对 pose/gripper 使用 [::self.stride]。

wyt 的 mapped_5hz_windows 字段表示观察起点的采样密度，不能解释成动作也是 5 Hz。两种设计服务不同的动作时间尺度，不能直接认定其中一种错误或性能必然更好。

## 5. 语言：HJZ 当前并未将 Luna 细粒度指令送给模型

HJZ 的 _WINDOW_COLUMNS 包含 task_index、annotation.valid，但没有 annotation.instruction 或 annotation.segment_index。read_window 通过 self.table.tasks[position] 生成 lang。即使源 parquet 带有 Luna 标注，当前实现也不会自动采用它。

wyt 后期设置：

~~~yaml
language_source: luna_fine
language_window_policy: same_segment
~~~

其 loader 返回 anchor 行的 annotation.instruction。same_segment 对应的离线索引约束为：当前行和后续 16 行均有效，instruction 相同，segment_index 相同，curated 起止区间相同；即使两个不同片段碰巧使用相同文字，也不允许跨片段。

**重要实现边界：**正式训练包设置 audit_window_contract=false。运行时信任已构建的同片段索引及其声明，读取当前行文字，并不在每次 __getitem__ 中重新逐行执行完整的同片段审计。umi_language.py 中的严格校验函数说明完整约束，但不能据此声称训练时每个 batch 都重新检查了一遍。

episode 级任务描述也可以是合法的训练目标。这里发现的是监督粒度不同，不是仅凭文字不同就判定某个标签错误。

## 6. 真实 UMI 窗口抽查

在 starvla-pai-check 容器内，使用 HJZ 当前 loader 及正式 UMI 根目录：

- 固定随机种子 42，均匀抽取 32 个候选窗口。
- 不解码图像，不执行模型前向或训练。
- 在本次独立进程的内存中补充读取 annotation.instruction、annotation.segment_index；没有修改源代码或源数据。
- 保存证据到 docs/umi_dataloader_sample_audit_20261020.json。

| 观测 | 结果 |
|---|---:|
| 成功读取的窗口 | 32/32 |
| 当前 loader 返回的文字与 anchor Luna 指令逐字相同 | 0/32 |
| 17 个选中时刻处于同一标注片段 | 21/32 |
| 选中时刻跨越标注片段 | 11/32 |
| 连续 97 行中出现 annotation.valid=false | 0/32 |
| 实际动作时间跨度 | 约 3.2 秒 |

例子，episode 24348、anchor 2238：

- HJZ 返回：replace the insole of a white shoe on a red plastic stool
- 当前 Luna 指令：Continue seating and adjusting the insole inside the shoe.

另一个跨片段例子，episode 127240、anchor 414：

- HJZ 返回：replace the water in a small potted plant on a marble table
- 当前 Luna 指令：Pick up the potted plant from the shelf.
- 未来 3.2 秒内选中的动作时刻跨越了标注片段。

这些样本说明两套语言来源确实不同，并且当前窗口可以跨片段。32 个窗口不足以估计全数据的跨片段比例，也没有证明这些 episode 级任务描述语义错误。

## 7. 动作、四元数与归一化

### 7.1 后期的动作几何定义基本一致

每只手均使用以下 8 维：

    [R_t^T(p_future-p_t) 的 3 维, q_t^{-1}⊗q_future 的 4 维, future_gripper 的 1 维]

双手拼成 16 维，四元数顺序为 xyzw。夹爪来自未来 magnetic_encoder 信号，保留连续值，没有变成夹爪增量或二值开关。

区别在于未来时刻集合，以及四元数符号选择。

### 7.2 四元数的数值标签不同

HJZ：

- quat_canonicalize 对每一个四元数独立找最大绝对分量，并使该分量为正。
- 同时用于当前 state 和未来相对四元数。
- 这能给 q 和 -q 选一个代表，但当最大分量切换时，不能保证沿轨迹连续。

wyt：

- 离线计算 episode_continuous_xyzw_v1 的符号表。
- 相邻四元数点积为负时翻转后一帧；在 episode 开头或无效/非连续数据处重置。
- 相对旋转计算保留这条连续的源符号路径，不对每个未来目标再次独立做半球翻转。
- 具体运行配置关闭 verify_quaternion_continuity，表示不逐样本再次做该诊断，不表示没有应用离线符号表。

q 和 -q 表示相同物理旋转，但对逐维 MSE 是不同标签。因此这个差异会改变损失的数值性质。尚未量化 HJZ 全数据中符号切换的实际频率。

### 7.3 归一化方法和统计量拟合范围都不同

HJZ：

    x_norm = clip(2*(x-q01)/(q99-q01)-1, -1, 1)

- 所有 16 维均使用相同类型的逐维变换，包括 quaternion、gripper。
- 统计文件抽样 200,000 个有效窗口，每个 action 窗口的 16 个目标均参与统计。
- 统计元数据明确是 horizon=16、frame_stride=6、anchor_stride=6。
- 文件里的 mean/std 仅用于诊断，训练实际不使用它们。

wyt：

    x_norm = (x-mean)/scale

- scale 通常为总体标准差；若小于设定阈值则按规则使用 1。
- 不执行 q01/q99 裁剪。
- 后期 relative/statistics_cached.json 的拟合来自训练池全部 947,780,205 个密集合法窗口。
- 拟合时 anchor stride=1，未来 action stride=1；state 计数 947,780,205，action 计数 15,164,483,280。
- 正式统计文件绑定数据指纹、训练视图、动作表示和允许应用的阶段，防止误用另一套统计量。

例如第一只手的相对位移 std，HJZ 统计文件约 [0.0859,0.0786,0.0661]，wyt 约 [0.0295,0.0253,0.0212]。这是各自物理目标的统计量，不是归一化后的 std；差异同时受到时间跨度、样本选择、符号和统计方法影响，不能只归因于其中一个因素。

**不能将 wyt 的统计 JSON 直接用于 HJZ 的 3.2 秒动作窗口。**

## 8. 数据池、验证集和五阶段

HJZ 原始两次 UMI 实验保存的 provenance：

- 320,712 个 episode。
- 160,269,278 个候选窗口。
- 一个训练 loader，无该 UMI loader 内单独的 episode 验证集。
- kept membership 来自 rational 清洗后的 selection/kept_episodes.parquet。

wyt 后期索引记录来自同一个 rational 清洗池的选择关系，并进一步：

- 剔除明确列出的 20 个 episode。
- 保留 701 个 episode 作为约 20 小时的验证池。
- 训练池剩余 319,991 个 episode。
- 计数满足 319,991 + 701 + 20 = 320,712；此关系与索引 lineage 一致，本次没有重新计算全数据 payload 的内容哈希。
- 训练密集合法窗口为 947,780,205；以有效 range 开头为相位、anchor stride=6 后，五阶段合计 159,198,571 个采样窗口。

| 阶段 | episode 数 | stride=6 后的窗口数 | global batch=2048 时预算 |
|---|---:|---:|---:|
| stage_01 | 63,998 | 31,842,655 | 15,548 updates |
| stage_02 | 64,000 | 31,838,251 | 15,546 updates |
| stage_03 | 64,001 | 31,842,015 | 15,547 updates |
| stage_04 | 63,995 | 31,836,896 | 15,545 updates |
| stage_05 | 63,997 | 31,838,754 | 15,546 updates |
| 合计 | 319,991 | 159,198,571 | 77,732 updates |

这表示按计划依次使用五个分片。没有证据表明这五个阶段是按任务难度排序的课程学习。

wyt 后期还配置了固定 512 个窗口、461 个 episode 的 validation_monitor。若要拿 HJZ 现有预训练模型去评测 wyt 的 UMI 验证集，必须注意 HJZ 的训练池包含原清洗池全部 episode，不能自动声称该验证集对 HJZ 是未见数据。这个问题不等于 LIBERO 下游验证集发生重叠。

## 9. 索引、异常处理与实际样本分布

| 项目 | wyt | HJZ |
|---|---|---|
| 样本目录 | 离线合法 range + cumulative + SQLite | episode 表 + 按长度推导候选 anchor |
| 起点约束 | 数据合法性与同片段等规则预先筛选 | 行读取时验证 selected frames 的 valid、unit quaternion、task_id 等 |
| 时间戳检查 | 离线窗口规则包含 timestamp gap 等约束 | read_window 检查时间戳有限，但没有同等的相邻时间差阈值检查 |
| 失败行为 | 读取失败抛异常，带 dataset index；不自动换样本 | 默认 strict=false，最多尝试原样本+20 个确定性随机替代候选 |
| batch 结构 | List[dict]，state/action/image/lang | 相同，并返回全 True 的 action_mask |
| 序列尾部 | 只使用完整 16 目标窗口 | 同样要求完整窗口，不依赖尾部 padding |

HJZ 的逻辑候选索引并不完全等于最终消费的样本流：解码或有效性检查失败时会替换窗口。替换是按 seed、epoch、原 index、attempt 确定的，但可能引入重复、跳过候选，并使难读样本的实际权重降低。当前没有统计真实训练期间的替换率，不能断言这一机制已造成明显偏差。

附带发现：HJZ 的 _load_cache 没有恢复 kept_filter_applied 属性，因此从缓存加载后的 provenance 可能显示 kept_filter=false；旧 UMI run 的记录为 true，最新 smoke 的记录为 false，但三者 episode 数都为 320,712。这首先是来源记录不完整，不能据此断言缓存绕过了 episode 过滤。

## 10. 图像读取与性能配置

共同点：

- 当前时刻三视角，顺序均为 head_left、wrist_left、wrist_right。
- PyAV 解码 RGB，再 resize 到 224×224。
- 没有在这些 loader 路径中看到随机裁剪、颜色增强或多帧视觉历史。

不同点：

| 项目 | wyt 后期 | HJZ 当前 |
|---|---|---|
| 视频根目录 | 原始 video_root 单独配置，支持显式根映射 | 从数据根目录下 videos/ 解析 |
| 普通匹配容差 | 1/30 秒 + 1e-6 | 1/30 秒 |
| episode 首时刻 | 配置约 2/30 秒的较大容差 | 无单独首时刻容差 |
| PTS 边界 | 按逻辑区间筛选候选，允许配置的 1ms slack；记录使用情况 | 检查查询 target 在区间内，未对每个候选 PTS 应用同样边界筛选 |
| Parquet 缓存 | 每 worker 512 MiB、最多128个row cache条目、32个文件句柄 | 默认128 MiB、16个文件句柄 |
| 视频句柄缓存 | 每 worker 64 | 默认每 worker 8 |
| 正式配置 workers | 每 rank 2 | 每 rank 4 |
| multiprocessing | spawn | fork |
| prefetch_factor | 2 | 2 |
| 批量视频解码开关 | 支持该功能，但本次正式计划为 false | 逐样本、逐视角解码 |

不能因为 wyt 有 umi_video_batch.py，就声称正式训练已经启用了批量解码。

两套实现的进程入口结构不同。HJZ 代码已有注释说明 spawn 会重新导入顶层 Accelerator 初始化；因此不建议仅将 fork 改成 spawn 来照搬 wyt 的 worker 设置。

## 11. 随机采样、分布式与恢复

- 两者都做“块顺序随机 + 块内随机”，不是全数据规模的完全随机排列。
- HJZ block_size=4096；wyt 正式方案 block_size=512、interleave_blocks=1。wyt 代码支持交错多个远距离块，但所查正式方案没有开启。
- 两者通过 Accelerate 对 loader 分片，不是再额外套一个手工 DistributedSampler。
- 在常见正式配置中，wyt 64 rank × 每 rank32 = global2048；HJZ 原两组为 64×24=1536。Oct19 的 world8 变体为 global256，应与64卡方案区分。
- WYT StageLoader 明确记录 stage、epoch、已提交全局 sample cursor，恢复时将 sampler 定位到 cursor，再恢复 rank RNG/loader generator。
- HJZ 当前 strict resume 记录 batches_consumed 和 epoch，重建 iterator 后逐 batch 快进。自己的 sampler 虽然也提供 start_index API，但当前 trainer 恢复路径并未像 wyt 一样直接用它定位。

这些差异主要影响样本顺序、吞吐和恢复成本；它们本身不能解释特定一次前向中的数千 MSE。

## 12. 不能混淆的模型端差异及历史版本

这些不属于 loader 算法，但决定 loader 输出是否真正进入训练：

- wyt 后期使用 state；构造模型时 include_state=True，state encoder 将当前 state 与未来 token、动作 token 一起送入 DiT。
- HJZ 原 UMI 配置 use_state=false；loader 虽然读出并归一化 state，模型不使用它。
- wyt 正式 repeat=1；HJZ DualStreamQwenPI 代码当前将 repeated_diffusion_steps 强制设为 2。这会重复同一图像条件并采样两组 flow matching 随机变量，不表示多读了一条不同的 UMI 样本。
- 两者模型还有普通 QwenPI/双流视觉、精度路径等差异。不能将所有 loss 差异归于 dataloader。

历史版本尤其重要：

| wyt 实验 | 动作表示 | 时间标签 |
|---|---|---|
| Oct14 R1 / 2000 steps 系列 | 配置未指定 action_pose_repr，默认 world 绝对动作 | 连续未来16帧，约0.533秒 |
| 后期 clean20/sameprompt/relative 系列 | current_eef_relative | 同样连续未来16帧，约0.533秒 |

之前讨论中引用的早期成功训练曲线，不能当成“与 HJZ 相同相对动作、相同时间跨度，只多了两处 LayerNorm”的严格对照。

## 13. 修改建议

1. **先确定目标时间尺度。**若目标是复现 wyt，使用 action frame stride=1、anchor stride=6，并重新拟合对应统计量。若目标确实是预测未来3.2秒，保留当前时间定义，比较时让双方模型使用同样的标签。
2. **明确语言任务。**如果想利用 Luna 的细粒度标注，应显式读取 annotation.instruction，并定义是否要求动作窗口不跨片段。若保留3.2秒窗口，不能直接照搬wyt的17行检查，应针对97行跨度或明确定义的目标集合重建规则，评估可用窗口损失。
3. **单独对照四元数符号策略。**先对实际轨迹统计 canonicalize 前后的相邻点积/符号切换率，再评估 episode 连续符号。改变策略后重新生成统计量。
4. **归一化做受控实验。**mean/std 与 q01/q99 没有仅凭代码就能确定的优劣。保持标签、数据划分和训练设置一致，比较验证集 loss 和物理空间误差；不要直接比较标准化坐标中的原始 loss。
5. **建立 episode 级独立验证池。**先划分再拟合统计量；保留固定验证窗口及数据指纹，以便判断修改是否提高了泛化。
6. **量化读取失败与替换。**在正式训练日志按 worker 汇总替换次数/原因，区分数据无效、PTS边界、损坏视频和资源问题；这比只看总吞吐更能解释实际消费分布。
7. **最后再调性能。**缓存大小和worker数应根据NAS延迟、CPU内存和吞吐测量决定；不要把纯性能调整和标签语义调整混在同一实验。

本次没有直接实施以上建议，以免在调查过程中改变现有训练任务。

## 14. 关键代码定位

下列 WYT 相对路径位于 umi_world8_zero2_dlc_20261019_v1 下，五个关键 loader 文件已与 Oct20 续训入口引用的冻结 ZIP 核对一致。

| 主题 | HJZ | WYT |
|---|---|---|
| 样本/时间标签 | starVLA/dataloader/umi_lerobot/umi_dataset.py:300 | code/starVLA/dataloader/umi_indexed_dataset.py:438 |
| anchor索引 | starVLA/dataloader/umi_lerobot/episode_index.py:252 | code/starVLA/dataloader/umi_indexed_dataset.py:273 |
| 动作坐标变换 | starVLA/dataloader/umi_lerobot/umi_dataset.py:365 | code/starVLA/dataloader/umi_pose.py:117 |
| 归一化 | starVLA/dataloader/umi_lerobot/normalize.py:72 | code/starVLA/dataloader/umi_normalization.py:446 |
| 语言定义 | starVLA/dataloader/umi_lerobot/umi_dataset.py:345 | code/starVLA/dataloader/umi_language.py；umi_indexed_dataset.py:489 |
| 视频解码 | starVLA/dataloader/umi_lerobot/umi_dataset.py:388 | code/starVLA/dataloader/umi_indexed_dataset.py:514；umi_video_time.py |
| 异常替代 | starVLA/dataloader/umi_lerobot/umi_dataset.py:430 | code/starVLA/dataloader/umi_indexed_dataset.py:553 |
| loader构造 | starVLA/dataloader/umi_lerobot/__init__.py:39 | code/starVLA/dataloader/umi_indexed_dataset.py:789 |
| 随机采样 | starVLA/dataloader/umi_lerobot/umi_sampler.py | code/starVLA/dataloader/umi_sampler.py |
| 阶段/恢复 | starVLA/training/train_starvla.py:371 | code/starVLA/training/trainer_utils/umi_training_data.py:174 |

样本抽查证据：docs/umi_dataloader_sample_audit_20261020.json。


## 15. 最新实际运行版本与 state 路径复核（2026-10-20）

用户追问后，进一步核对了实际运行任务，而不只依赖实验目录名：

- 当前仍更新日志的 UMI attempt：resume-10000-to-77732-20261020-v1；本次查看的日志已记录 global update 18876。
- 启动入口：umi_resume_10000_77732_20261020_v1/launch_resume_dlc.sh。
- 此入口解包 umi_clean20_recovery_meanval_4x16_19000_20261015_v1/frozen_bundle.zip，并断言完整包 SHA256。
- 文件实际 SHA256 与入口固定值相同：f692cb1c1e3898011034d4e961eab75024bf776d32136ce375146c6387e4a211。
- ZIP 内 input_plan.json 的 SHA256：31a47613300b9c2426f3b93386c41796f6d0aca3e49beeedbc0c164441c29901。
- 实际 attempt 的 startup_rank_0.json 所记 source_plan_sha256 与上述值完全一致，runtime 记录为 world_size=64、batch_size=32、global_batch_size=2048。
- 该 attempt 的记录位于原实验 runs/training/attempts/resume-10000-to-77732-20261020-v1；实时日志位于 runs/launches/同名attempt/node_0/training.log。

因此，本报告所查的是这条当前 UMI 续训真正绑定的冻结 loader 版本。冻结包日期为 Oct15 不意味着 Oct20 任务换成了另一套 loader。当前可见的 Oct20 更晚 LIBERO posttrain 启动文件属于下游训练，不属于这条 UMI 预训练。

state 参与训练的代码证据均直接来自上述 ZIP：

1. umi_indexed_dataset.py:505：state = values[:1]，即当前观察时刻的两手绝对世界位姿与夹爪，形状 [1,16]；future actions 是 values[1:]，两者已区分。
2. 同文件 :729：state 经过固定 mean/std 标准化。
3. train_umi_pretrain.py:143-145：构造模型配置显式 include_state=True。
4. QwenPI.py:212、239-248：从 examples 读取 state，转为 FP32，并作为第三个参数传给 action_model。
5. LayerwiseFM_ActionHeader.py:255：根据 state_dim=16 构建 state_encoder。
6. 同文件 :353、363-366：计算 state_features，并沿 token 轴拼入 state/future/action 序列。
7. DiT 使用交替 self/cross attention，动作 token 可以在 self-attention block 中读取 state token。
8. run_training64_zero2.py:313-316：原样将 batch 交给 engine，再对 action_loss 反向传播；没有删除 state 的包装逻辑。

这里使用的是当前状态，不是未来真实状态。state 通过独立的 MLP 进入 DiT。

证据边界：本次 startup 的 compute_dtypes 为空、runtime_validation_enabled=false；其中 expected_executed_modules 字段是预期列表，不能作为本次实际 hook 测量。state 参与计算的结论来自实际任务绑定的代码、配置和完整调用链，而非伪称获取了本次 state encoder 梯度实测。
