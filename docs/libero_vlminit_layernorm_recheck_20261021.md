# VLM-init 与随机初始化 UMI 模型的 LIBERO 后训练复核及 LayerNorm 重跑

日期：2026-10-21；开发机：`ppu-dev`；测试与训练容器：`starvla-pai-check`。

## 结论

**旧日志支持“VLM-init 组长期高位抖动，而随机初始化预训练组较低且持续下降”。** 更准确地说，VLM-init 组也有很小的改善，并不是数值完全不降；但与随机初始化预训练组相比，收敛明显停滞。

这里的“随机初始化组”指其 UMI 预训练起点没有继承 VLM 权重；LIBERO 后训练同样加载了 UMI 预训练 checkpoint，不是随机模型直接做 LIBERO。

**曾从原 VLM-init UMI 98k checkpoint 新开 LayerNorm 后训练实验并启用 W&B；该 run 现已按用户要求停止，见第10节。** 是否改善应观察新 run 的趋势；不能仅凭加入模块就宣布问题解决。

## 1. 检查了哪些证据

两个旧 run：

- `playground/Checkpoints/multiview_libero_vlminit_umi98k_gb256_backup`
- `playground/Checkpoints/multiview_libero_umi110k_gb256_backup`

对每个 run 读取：

1. `config.full.yaml`：递归逐字段比较。
2. `wandb/wandb/run-*/files/output.log`：解析全部带 step 的 action loss、grad norm、实际学习率、模型时间。
3. `wandb-summary.json`：核对末次 loss/step 与文本日志一致。
4. `dataset_statistics.json`：对比 SHA256，验证两组导出的归一化统计相同。
5. 检查实际可用的 UMI 98k/110k checkpoint 路径、当前 LayerNorm 实现、配置开关和加载兼容性。

日志是 rank 0 的 batch loss，每10个 optimizer steps记录一次，并非所有卡的全局平均。下表对比相同 optimizer-step 区间的记录均值，避免把随机组更长的训练尾段当成同训练量的比较。

## 2. 相同步数下的 loss

| 记录区间 | VLM-init loss 均值 | 随机初始化预训练组 loss 均值 |
|---|---:|---:|
| 10-100 | 1.258519 | 0.149130 |
| 10-500 | 1.250902 | 0.080479 |
| 501-1000 | 1.254423 | 0.045627 |
| 1001-2000 | 1.242279 | 0.031234 |
| 2001-3000 | 1.230792 | 0.019509 |
| 3001-4000 | 1.219634 | 0.013928 |
| 4001-5000 | 1.208629 | 0.011182 |
| 4680-5670 | 1.213321 | 0.008461 |

关键点：

- 第10步：VLM-init **1.217677**；随机初始化预训练组 **0.260388**。
- 共同末段4680–5670：VLM-init **1.213321 ± 0.044140**；随机初始化预训练组 **0.008461 ± 0.003754**，前者约为后者 **143.4倍**。±表示这些日志记录的标准差，不是置信区间。
- VLM-init 由最初10–100步均值1.258519到共同末段1.213321，仅降低约3.6%；随机组由0.149130到0.008461，降低约94.3%。
- VLM-init 最后记录为5670步、loss1.195502；随机组继续训练到18530步、最后loss0.004546。后者只作为长程结果，不与VLM-init末步直接做等训练量比较。
- 两份日志都以 KeyboardInterrupt 结束，现有记录不代表各自配置的训练总步数已完成。

图中上半部分为共同窗口loss（淡点为原始记录，曲线为20条记录的移动平均，纵轴对数），下半部分为实际学习率：

![旧实验共同窗口](../playground/libero_ln_recheck_20261021/old_runs_common_window.png)

## 3. 梯度范数说明

共同末段4680–5670的记录中，VLM-init的梯度范数中位数为 **0.156072**，随机组为 **0.500213**。

高loss、较小梯度和很弱的下降趋势，与优化困难相符；梯度范数不是0，也不能据此单独证明“完全没有学习”“恒等预测0”或“attention sink就是唯一原因”。这些日志没有保存旧模型的逐token激活或预测张量。

## 4. 对照条件及重要混杂因素

### 两组相同项

- 实际 DeepSpeed 审计：ZeRO-2、bf16、每卡16、16卡、累积1、global batch256。
- 模型结构、28层DiT、repeat=2、use_state=False、16步动作窗口。
- LIBERO数据混合、相对EEF动作表示、统计文件和数据设置。
- seed42、全参数训练、VLM学习率1e-5、action_model学习率1e-4、AdamW及梯度裁剪。
- 导出的统计文件 SHA256 相同：`fa9b999aac06a0577eea3af78f032ad15a1726b8177a8844bcf5bdc927d2d12e`。
- 旧保存配置没有开启本次新增的两处归一化开关；当前实现默认值为none。

### 两组确实不同的项

递归比较完整配置，仅发现run/output名称、UMI起始checkpoint和 `trainer.max_train_steps` 不同：VLM-init是UMI98k、max_train_steps=30000；随机组是UMI110k、max_train_steps=100000。

所以预训练步数也不完全一致，不能把差异全部视为初始化方式的严格因果效应。

### 已发现旧学习率进度异常

两组YAML均为warmup5000，但第10步的action_model实际学习率是 **3.2e-6**。按每次optimizer更新推进一次，应为：

```text
1e-4 × 10 / 5000 = 2e-7
实际 / 预期 = 16
```

这与旧版Accelerate scheduler按16个分布式进程推进相符；当前trainer已使用 `step_scheduler_with_optimizer=False` 并仅在实际更新边界调用scheduler。

旧warmup实际约在312.5个optimizer更新时完成，而不是5000。旧两组总调度长度又分别为30000和100000，因此它们的实际学习率在warmup后明显不同；VLM-init记录中还出现了学习率降低后再次升高。不能把这视为计划内、完全一致的训练条件。

**本次使用已修正的trainer，不重现该错误。与旧run的比较是“当前代码 + LayerNorm的实用修复验证”，不是只改变LayerNorm的严格消融。** 严格消融应再用同一当前代码、同一checkpoint、同一调度和batch运行norm=none对照；本次未额外启动这组对照。

## 5. 本次新实验

| 项目 | 配置 |
|---|---|
| run_id | `multiview_libero_vlminit_umi98k_ln_gb256_2ppu_20261021_165426_9de0ca` |
| 起始权重 | `starvla_umi_b24_vlm_init/checkpoints/steps_98000_pytorch_model.pt` |
| 运行位置 | ppu-dev / starvla-pai-check |
| 拓扑 / batch | 2 PPU × 每卡16 × 累积8 = global256 |
| repeat | 2，与旧LIBERO实验一致；UMI正式预训练的repeat=1不覆盖此项 |
| state | False，与旧LIBERO实验一致 |
| VLM初始化字段 | load_weights=False；模型构建后加载完整UMI98k checkpoint，保留其VLM权重，不是丢弃VLM权重 |
| 两处LayerNorm | cross-attention K/V条件输入前；action decoder输入前 |
| LayerNorm参数 | elementwise_affine=False，无新增可训练参数 |
| 长度 / warmup | 15000 / 750，当前正式LIBERO配置 |
| 学习率 | VLM1e-5，action_model1e-4，base3e-5 |
| 精度 / ZeRO | bf16 / ZeRO-2 |
| 梯度检查点 | False |
| 保存 | 每5000步保存完整状态和模型权重，保留3个周期checkpoint |
| eval | 间隔10000000，本轮无周期eval |
| 日志 | 每optimizer step记录，另有每rank/microbatch显存与loss telemetry |
| W&B | 用户已明确授权，SDK验证保存登录有效，online模式 |

新run目录：`/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Checkpoints/multiview_libero_vlminit_umi98k_ln_gb256_2ppu_20261021_165426_9de0ca`。

此次仅给已有 `examples/simBenchmarks/LIBERO/train_files/starvla_dualstream_libero.yaml` 增加归一化和明确repeat配置；未新增训练YAML。checkpoint、累积和run_id通过CLI指定。旧两组目录和权重未覆盖。

## 6. 做过的测试和运行审计

### CPU预检：通过

使用真实action head构造小维度两层模型：

- LayerNorm开启/关闭前后state_dict键完全一致。
- 旧head权重可strict=True加载到新增LayerNorm的head。
- 条件特征中加入绝对值14784的分量，启用LayerNorm的forward/backward loss与梯度均有限。
- 新实验入口已在规定容器内编译通过。

这证明配置及参数兼容性，不证明完整训练必然收敛。

### 完整训练内审计

新入口沿用普通trainer，仅增加观测与断言：

- 实际构造的decoder及所有cross blocks必须是无仿射参数LayerNorm。
- 记录完整UMI checkpoint加载后的missing/unexpected keys；非空即停止，防止静默忽略权重。
- 首次前向记录两处归一化前后首个向量的absmax/RMS/均值/std。
- 首次head输入应为16×repeat2=32；这不是dataset batch翻倍。
- 首次记录loss与零预测高斯噪声期望基线 `1 + mean(action²)`（使用相同有效action mask）；该量只是期望基线，不能反推旧模型的预测。

## 7. 如何判断结果

优先观察100、500、1000步的区间loss均值与分位数，而不是只看首步。加入归一化改变了动作头输入分布，首步loss可能不低；关键是后续是否能显著脱离旧的约1.2平台、保持有限梯度、且训练稳定。

若loss下降改善，可认为当前修复组合有效；若仍停滞，需要在同一代码条件下做norm=none对照、检查输出与目标、数据归一化、条件特征和VLM/DiT分组梯度。

## 8. 证据与复现文件

- `playground/libero_ln_recheck_20261021/history_comparison.json`
- `playground/libero_ln_recheck_20261021/vlm_init_history.csv`
- `playground/libero_ln_recheck_20261021/random_init_history.csv`
- `playground/libero_ln_recheck_20261021/old_runs_common_window.png`
- `playground/libero_ln_recheck_20261021/layernorm_preflight.json`
- `playground/libero_ln_recheck_20261021/new_run.json`
- 新run内 `command.json`、`code_snapshot/`、`code_sha256.json`、`training.log`、`run_status.json`、`telemetry/`。

## 9. 新实验启动后的实测快照

时间：2026-10-21T17:01:09.370258；状态：running；已完成16个optimizer updates。

W&B：[本次run](https://wandb.ai/hejz26-tsinghua-university/qwenpi-vla/runs/cxyl4z5l)。

实际DeepSpeed审计：`zero_stage=2 micro_batch=16 grad_accum=8 train_batch=256`，bf16启用。两卡均记录到完整checkpoint加载的missing/unexpected keys为空。

| optimizer step | rank0最后microbatch loss | rank1最后microbatch loss |
|---|---:|---:|
| 1 | 1.227699 | 1.260290 |
| 2 | 1.301896 | 1.388122 |
| 3 | 1.238448 | 1.207434 |
| 4 | 1.240523 | 1.233685 |
| 5 | 1.265037 | 1.317506 |
| 6 | 1.327353 | 1.272268 |
| 7 | 1.268856 | 1.361351 |
| 8 | 1.302394 | 1.232857 |
| 9 | 1.258964 | 1.292231 |
| 10 | 1.193058 | 1.266530 |

此表不是8个microbatch或全部卡的平均loss，仅为每次optimizer更新最后一个microbatch的原始loss，口径与trainer日志一致。最初若干步仍在1.2–1.3附近，尚无证据宣布收敛改善。

- rank0：截至快照的PyTorch已分配峰值67.45 GiB，观测到的设备占用最高79.62/96 GiB。
- rank1：截至快照的PyTorch已分配峰值67.55 GiB，观测到的设备占用最高79.61/96 GiB。

首次归一化前后测量仅选择rank0首样本中的一个向量，不是整层全token最大值：

- first_cross_condition：RMS 0.243962 → 0.999916；absmax 2.343750 → 9.589464。
- action_decoder_input：RMS 740.100403 → 1.000000；absmax 2684.361816 → 3.635470。
- 首次head输入batch=32，masked loss=1.243739，相同动作标签下零预测的高斯噪声期望loss约1.269009。这不能证明旧模型恒为零预测。

学习率复验：第16步action_model LR=2.13333333333e-06，按warmup750预期为2.13333333333e-06；本次调度未按world_size额外加速。

具体机器可读证据：新run的 `startup_verification.json` 和 `telemetry/`。此为当时的运行快照；该 VLM-init run 现已按用户要求停止，实际完成49次更新，见第10节。

## 10. 用户要求切换为随机初始化 UMI 预训练组

更新时间：2026-10-21T17:12:36.652667。

- 原 VLM-init+LayerNorm run `multiview_libero_vlminit_umi98k_ln_gb256_2ppu_20261021_165426_9de0ca` 已按用户要求停止，完成49次更新；rank0最后一个同步microbatch loss=1.288127，日志保留。
- 新 run：`multiview_libero_randinit_umi110k_ln_gb256_2ppu_20261021_170934_3a9a57`。
- 加载完整随机初始化 UMI 预训练组的 `starvla_umi_b24/checkpoints/steps_110000_pytorch_model.pt`，不是随机模型直接做LIBERO。
- 两处LayerNorm继续开启；双卡×每卡16×累积8=global256，repeat2、state=False、15000步、warmup750、学习率及其余设置保持一致。
- W&B继续使用用户已授权的项目及登录。
- 最新活动run的查询入口：`playground/libero_ln_recheck_20261021/active_posttrain_run.json`。
- 新run目录：`/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Checkpoints/multiview_libero_randinit_umi110k_ln_gb256_2ppu_20261021_170934_3a9a57`。
