# LIBERO 15k 评测两极分化与原因核查

日期：2026-10-23；主机：ppu-dev；复核在 starvla-pai-check 容器内完成。

## 结论

两极分化已确认。优先问题是把未来实测夹爪宽度经全局41.932mm阈值转换为开闭命令：持物宽度超过阈值时，正确的持物状态也会被误译成张开。Object 的失败分布与此高度吻合，Long 的4个零成功任务也都涉及这些物体。该问题不能解释全部Goal失败，也尚无在线A/B证明修复后完整成功率。

## 数据与核验范围

- checkpoint：playground/Checkpoints/multiview_libero_umi110k_gb256_backup/checkpoints/steps_15000_pytorch_model.pt
- 原始评测：同run的eval_15k_logs/libero_{spatial,object,goal,10}.log，2026-10-19至21日。
- 逐条解析40任务×50次的Success记录，与每任务最终成功率以及2000个results目录下视频文件名的success/failure后缀交叉核对，全部一致。未逐一人工观看2000条视频。
- 独立重新读取1693条示范的parquet state/action，覆盖四套件所有任务；夹爪统计与既有playground/Evaluation/libero_diagnosis_20261023/expert_gripper_audit.json一致。
- 稳态定义：同一源开闭命令连续8帧；用action[t]与state[t+1]宽度比较。统计为帧加权，不等同于rollout成功率或失败概率。
- Spatial日志末尾确有launcher引号错误，但发生在500次完整评测之后；status TSV亦明确记录completed_500_episodes_488_successes，不能将该尾部错误解释成低成功率原因。

## 套件结果

| 套件 | 成功/次数 | 成功率 | ≤10%任务数 | ≥80%任务数 | 中间任务数 |
|---|---:|---:|---:|---:|---:|
| libero_spatial | 488/500 | 97.6% | 0 | 10 | 0 |
| libero_object | 166/500 | 33.2% | 6 | 3 | 1 |
| libero_goal | 311/500 | 62.2% | 3 | 6 | 1 |
| libero_10 | 224/500 | 44.8% | 4 | 4 | 2 |

均分59.45%。非Spatial共30个任务中26个落在≤10%或≥80%两端。

官方StarVLA-π(Qwen3-VL) README的30k结果依次为Spatial98.8%、Object99.6%、Goal95.8%、Long88.4%，只作参考，不是同结构同训练量对照。

官方来源：https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/README.md

官方checkpoint card还给出10k的89.6/85.0/75.2/46.0，以及20k的95.2/88.6/83.4/63.6（Spatial/Object/Goal/Long顺序）。这说明训练阶段会影响差距，但Object在早期也没有本次那么低。

Card：https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1/blob/main/README.md

## 首要原因：持物宽度被当作开闭意图

训练标签由examples/simBenchmarks/LIBERO/train_files/data_registry/data_config.py的libero_state_windows_to_umi16构造：第8维是未来实际两指关节位置之差，而不是原数据中的二值夹爪命令。

deployment/model_server/policy_norm_processor.py以(q01[7]+q99[7])/2得到41.932mm；examples/simBenchmarks/LIBERO/eval_files/umi16_to_osc.py对width>threshold输出-1（张开），否则+1（闭合）。已直接调用真实转换函数验证：53.5mm输出张开，33.6mm输出闭合。原评测policy_server.log也记录了相同阈值。

归一化范围中点不是开闭类别分界。夹爪受物体接触约束时，持续闭合命令可以对应很大的实际宽度。这使低动作回归loss和低任务成功率能够同时出现，甚至准确预测示范宽度仍可能得到错误的控制命令。

| Object任务 | 成功率 | 稳态闭合宽度中位数(mm) | 闭合帧被旧阈值判为张开的比例 |
|---|---:|---:|---:|
| ketchup | 100% | 33.65 | 2.2% |
| bbq sauce | 92% | 36.34 | 5.7% |
| salad dressing | 94% | 36.80 | 0.1% |
| butter | 44% | 38.94 | 2.3% |
| cream cheese | 2% | 42.26 | 87.7% |
| chocolate pudding | 0% | 45.90 | 99.8% |
| orange juice | 0% | 53.48 | 98.4% |
| milk | 0% | 53.53 | 99.3% |
| alphabet soup | 0% | 62.42 | 96.2% |
| tomato sauce | 0% | 63.40 | 98.0% |

Object逐任务成功率与该误开比例的Pearson相关系数为-0.941（仅10任务，相关性本身不是因果证明）。宽度会随抓取姿态而变化，表中是所有示范稳态闭合帧的中位数，并非物体固定尺寸。

Spatial的10个任务都抓黑碗，闭合宽度中位数约4.8–9.0mm，远低于41.9mm，因此基本绕过这个误转换区域。

Long的4个0%任务：alphabet soup+tomato sauce、cream cheese+butter、white mug+chocolate pudding、alphabet soup+cream cheese。这些恰好都包含Object中接近0%的物体。一本书任务100%，双杯86%，碗入抽屉84%，杯入微波炉80%，不能把Long低分简单理解为模型普遍不会长程。

Goal中cream cheese入碗2%也吻合该机制；其他异常还需独立解释。

## 已有固定图像预测记录：支持但不是在线因果实验

读取并汇总既有fixed_image_predictions.jsonl（本轮没有重新加载模型或生成这些预测）。每个物体仅一条示范、一个holding取样点、3个噪声种子；该取样点不保证已达到稳态。当前四个被记录源码文件与当时元数据SHA256一致情况：{"starVLA/model/framework/VLM4A/DualStreamQwenPI.py": true, "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py": true, "deployment/model_server/policy_norm_processor.py": true, "examples/simBenchmarks/LIBERO/eval_files/umi16_to_osc.py": true}

| 物体 | 示范未来宽度中位数(mm) | 模型宽度均值(mm) | 旧阈值模型动作张开比例 |
|---|---:|---:|---:|
| alphabet_soup | 39.54 | 39.27 | 0% |
| bbq_sauce | 37.77 | 37.35 | 0% |
| butter | 38.95 | 38.96 | 0% |
| chocolate_pudding | 45.86 | 45.08 | 100% |
| cream_cheese | 46.01 | 45.37 | 100% |
| ketchup | 33.70 | 33.86 | 0% |
| milk | 54.53 | 53.74 | 100% |
| orange_juice | 53.09 | 52.83 | 100% |
| salad_dressing | 36.81 | 37.06 | 0% |
| tomato_sauce | 63.79 | 63.30 | 100% |

这些点的源夹爪命令均为闭合。milk/orange juice/tomato sauce/chocolate pudding/cream cheese的预测宽度接近示范，旧阈值却将其全部解码为张开。alphabet soup此取样点宽度尚低于阈值，不能用单点推广其整个轨迹；全示范稳态审计才显示它96.2%的闭合帧超过阈值。固定图像结果不能证明闭环轨迹正确、目标选择正确或阈值修复后的成功率。

## 不能归给阈值的剩余失败与其他因素

1. Goal：open top drawer + bowl仅2%，其闭合帧误开率0.27%，明显无法由持物宽度误判充分解释；推盘子10%和开中抽屉34%的示范没有稳态闭合帧。接触状态下开命令也可能得到受约束的窄宽度，宽度本身仍不能可靠还原意图。需检查闭环轨迹、接触几何与策略阶段切换。
2. Object的butter44%，其稳态闭合误开率仅2.3%，还存在定位、接近抓取、过渡时机、姿态或预测误差等原因。具体哪项主导尚未证明。
3. 实测未来EEF轨迹不等同于原始OSC控制命令。当前用位置/姿态误差除以0.05m/0.5rad逆算OSC且裁剪，接触、推拉、跟踪滞后中不保证等价于示范动作。16步才更新图像，use_state=false，均可能影响接触与阶段切换；这是待消融假设。
4. 当前为DualStreamQwenPI、2B、UMI110k预训练后LIBERO15k、UMI16/horizon16/双相机。官方card是4B QwenPI，7D delta控制动作、horizon8等不同接口，README对比30k，不能视为同模型复现。load_weights=false是构建时随机初始化，再加载完整UMI权重；不能说评测用了随机未训练模型。
5. 既有docs/libero_vlminit_layernorm_recheck_20261021.md已记录旧scheduler按16进程推进导致warmup约312.5个optimizer updates完成，而非5000；这是训练条件的混杂因素，可能影响质量，但未证明是本次物体分化原因。
6. 该15k是旧随机初始化UMI预训练组，不是VLM-init高loss组，也不是10月21日新开的LayerNorm实验。旧配置两处norm默认none，不能将后来加LayerNorm的分布变化倒推成10月19日评测失败原因。

## 建议验证顺序

1. 固定15k权重、图像、初始状态和每个chunk的采样种子，仅改变夹爪解码，先测milk/orange juice/tomato sauce/cream cheese，再加ketchup与Spatial黑碗作回归控制；比较同一初态成对轨迹、抓住后掉落和最终成功。不能只比较两次不同随机噪声的总均值。
2. 70mm可作为诊断候选：离线稳态闭合帧上Object误开率近0（cream cheese0.074%），但它会误判部分原本张开的帧；例如Goal推盘子稳态开帧有33.8%会被判闭合。不能未经闭环验证就把70mm设为最终通用修复。
3. 长期更直接的方案是保留原二值夹爪命令作为监督，或以正确的连续目标宽度控制器执行宽度；重新训练时需明确动作接口，不能静默改变旧checkpoint语义。
4. 阈值问题排除后，单独对抽屉/推盘子/butter做源动作回放、示范未来位姿反解回放与模型闭环三者比较，再考虑缩短执行chunk和更长训练。

本轮未修改模型、训练、评测逻辑或checkpoint，也未启动训练或全量重评测。没有新在线A/B结果，不能给出修复后成功率。完整结构化复核数据见同目录libero_eval_bimodality_audit_20261023.json。

## 全部任务

| 套件 | ID（日志顺序，0起） | 成功/次数 | 成功率 | 任务 |
|---|---:|---:|---:|---|
| libero_spatial | 0 | 50/50 | 100% | pick up the black bowl between the plate and the ramekin and place it on the plate |
| libero_spatial | 1 | 50/50 | 100% | pick up the black bowl next to the ramekin and place it on the plate |
| libero_spatial | 2 | 50/50 | 100% | pick up the black bowl from table center and place it on the plate |
| libero_spatial | 3 | 50/50 | 100% | pick up the black bowl on the cookie box and place it on the plate |
| libero_spatial | 4 | 47/50 | 94% | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| libero_spatial | 5 | 48/50 | 96% | pick up the black bowl on the ramekin and place it on the plate |
| libero_spatial | 6 | 47/50 | 94% | pick up the black bowl next to the cookie box and place it on the plate |
| libero_spatial | 7 | 48/50 | 96% | pick up the black bowl on the stove and place it on the plate |
| libero_spatial | 8 | 50/50 | 100% | pick up the black bowl next to the plate and place it on the plate |
| libero_spatial | 9 | 48/50 | 96% | pick up the black bowl on the wooden cabinet and place it on the plate |
| libero_object | 0 | 0/50 | 0% | pick up the alphabet soup and place it in the basket |
| libero_object | 1 | 1/50 | 2% | pick up the cream cheese and place it in the basket |
| libero_object | 2 | 47/50 | 94% | pick up the salad dressing and place it in the basket |
| libero_object | 3 | 46/50 | 92% | pick up the bbq sauce and place it in the basket |
| libero_object | 4 | 50/50 | 100% | pick up the ketchup and place it in the basket |
| libero_object | 5 | 0/50 | 0% | pick up the tomato sauce and place it in the basket |
| libero_object | 6 | 22/50 | 44% | pick up the butter and place it in the basket |
| libero_object | 7 | 0/50 | 0% | pick up the milk and place it in the basket |
| libero_object | 8 | 0/50 | 0% | pick up the chocolate pudding and place it in the basket |
| libero_object | 9 | 0/50 | 0% | pick up the orange juice and place it in the basket |
| libero_goal | 0 | 17/50 | 34% | open the middle drawer of the cabinet |
| libero_goal | 1 | 50/50 | 100% | put the bowl on the stove |
| libero_goal | 2 | 50/50 | 100% | put the wine bottle on top of the cabinet |
| libero_goal | 3 | 1/50 | 2% | open the top drawer and put the bowl inside |
| libero_goal | 4 | 45/50 | 90% | put the bowl on top of the cabinet |
| libero_goal | 5 | 5/50 | 10% | push the plate to the front of the stove |
| libero_goal | 6 | 1/50 | 2% | put the cream cheese in the bowl |
| libero_goal | 7 | 48/50 | 96% | turn on the stove |
| libero_goal | 8 | 49/50 | 98% | put the bowl on the plate |
| libero_goal | 9 | 45/50 | 90% | put the wine bottle on the rack |
| libero_10 | 0 | 0/50 | 0% | put both the alphabet soup and the tomato sauce in the basket |
| libero_10 | 1 | 0/50 | 0% | put both the cream cheese box and the butter in the basket |
| libero_10 | 2 | 32/50 | 64% | turn on the stove and put the moka pot on it |
| libero_10 | 3 | 42/50 | 84% | put the black bowl in the bottom drawer of the cabinet and close it |
| libero_10 | 4 | 43/50 | 86% | put the white mug on the left plate and put the yellow and white mug on the right plate |
| libero_10 | 5 | 50/50 | 100% | pick up the book and place it in the back compartment of the caddy |
| libero_10 | 6 | 0/50 | 0% | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| libero_10 | 7 | 0/50 | 0% | put both the alphabet soup and the cream cheese box in the basket |
| libero_10 | 8 | 17/50 | 34% | put both moka pots on the stove |
| libero_10 | 9 | 40/50 | 80% | put the yellow and white mug in the microwave and close it |

## 后续WYT接口对照（2026-10-23）

见[wyt_libero_action_interfaces_20261023.md](wyt_libero_action_interfaces_20261023.md)。WYT早期使用预测宽度减当前宽度的闭环解码；新UMI-aligned使用执行器目标开度监督和连续controller，均不同于本报告所审计的HJZ41.9mm阈值。已重算其UMI20k新分支1956/2000（97.8%）；用户提及40k/91.60%具体run尚未定位。不能将阈值问题视为所有UMI16模型固有缺陷，也不能在HJZ闭环A/B之前断言它解释全部差距。
