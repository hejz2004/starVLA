# QwenPI 后段交替多视角注意力

本分支 `experiment/cross-view-architecture` 从原始 QwenPI 实现出发，移除了 DualStreamQwenPI、QWen3DualStream、旧 vendored 双流模型及专用调试测试。历史报告和 checkpoint 保留，不迁移旧权重。

## 实现

入口是 `framework.name: QwenPI`，动作头和 VLM 语言部分沿用 QwenPI。QWen3.py 选择新的 `qwen3_vl_cross_view` 模型；它直接基于 transformers 4.57.6 原始 Qwen3-VL 建立，保留原有参数名，并支持每个视角的 register tokens。当前 UMI/LIBERO 配方均启用每视角 4 个 registers。

```yaml
framework:
  name: QwenPI
  qwenvl:
    load_weights: true
    cross_view_attention:
      enabled: true
      start_layer: 16
      num_register_tokens: 4
```

层编号从 0 开始。24 层中 0–15 为视角内注意力，16/18/20/22 为跨视角注意力，17/19/21/23 为视角内注意力。`start_layer: null` 按模型深度取 floor(2*depth/3)。`num_register_tokens` 默认为 0，兼容历史配置；设为 4 启用本次改进。原始视觉编码器的精确对照需同时设 `enabled: false` 和 `num_register_tokens: 0`。仅关闭 enabled 会保留各视角内部的 registers。

本实现的后段从 global 开始交替，这是本实验明确选择的顺序。借鉴 DA3 的后段交替范围与全局位置处理，并非完整复制其几何网络、camera token、DPT 或几何监督。

跨视角层直接更新 patch hidden states，融合后的特征继续进入后续视觉层，并由原有 DeepStack / final merger 输出。没有 gate、独立交互投影或额外视觉 block。原有 DeepStack 的第 5/11 层输出仍为单视角特征，第 17 层输出及最终 merger 才包含视觉跨视角融合。

每个视角的局部序列为 `[4 个 register tokens | 本视角 patch tokens]`，两者在 local block 内双向交换信息。所有视角共享一份 `[4, vision_hidden_size]` 的可学习初始参数，每次前向展开为各视角独立的 token 状态；不同视角、不同样本的 register 状态各自演化，不按相机 ID 建立独立参数。新增参数仅为 `visual.register_tokens.weight`，参数量是 `4 * vision_hidden_size`。

global block 仅接收同一样本的 patch tokens，registers 完全绕过该 block（attention 与 MLP），原样交给下一层。后续 local block 可以通过已经融合的本视角 patches 间接接收其他视角的信息；registers 不直接参与跨视角 attention。DeepStack 和最终 merger 都先去掉 registers，只合并原有 patches，因此视觉输出数量、LLM image placeholders 和动作头输入长度不增加。Action DiT 原有的 `future_tokens` 与此处的 registers 是不同的参数。

packed 图像通过 input_ids 中的 vision_start/image token 推导每个样本的图像数量；global attention 按样本切分，local 按图像切分，支持混合视角数量和分辨率。文本样本可以没有图像。直接调用视觉塔需传 scene_image_counts，缺少边界会报错，不能把 batch 默认为一个场景。此实验仅支持静态图像视角；启用后的视频输入明确拒绝。

local 的 patches 使用原始二维 RoPE。registers 在加入图像绝对位置嵌入之后才插入，不添加任何绝对位置编码，其 RoPE 恒为 identity（cos=1、sin=0）。global 的 patches 也使用 identity RoPE（无二维相对旋转），不把不同相机像素坐标当成公共坐标；patch 原始绝对位置嵌入仍保留。不新增相机 ID 或外参。registers 仅支持静态图像视角。

UMI/LIBERO 的 action_mask 支持被保留到 QwenPI，含 diffusion repeat 对齐，避免把 LIBERO 的未使用右臂维度加入损失。支持 VLM gradient checkpointing。

## 配置与启动

- UMI：`examples/umi/train_files/starvla_qwenpi_da3_umi_v2.yaml`
- UMI 冒烟：`examples/umi/train_files/starvla_qwenpi_da3_umi_smoke.yaml`
- LIBERO controller-goal：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero_action_goals.yaml`
- LIBERO 原有 relative-EEF 变体：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero.yaml`
- LIBERO 冒烟：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero_smoke.yaml`

现有 UMI / LIBERO 启动脚本默认路径已更新。训练数据、动作定义、原有 state 设置与训练超参数保持原 YAML 选择：UMI state=true，LIBERO state=false。做统一无状态实验需显式把 UMI use_state 设为 false。UMI 默认读原始 VLM 预训练参数；完全随机初始化可覆盖 `framework.qwenvl.load_weights=false`。

所有 LIBERO 配置的旧 UMI checkpoint 默认路径已清空；正式配置默认从原始 VLM 初始化。新架构 UMI 训练完成后，通过 `PRETRAINED_CHECKPOINT=/path/to/new/checkpoint.pt` 调用 `run_libero_train.sh`。迁移时需使 action head 的 state 配置与目标实验一致。

包含 shared_init / interaction_stages 的旧双流 checkpoint 会明确报错，包括 strict=false 加载；旧 DualStreamQwenPI 配置不再受支持。加载原始 VLM 时保留全部原有权重，新增 register embedding 随机初始化；新 checkpoint 会保存该 embedding 及其配置。历史无 register 的配置仍不创建该参数，保持原有 checkpoint 结构。不要将新架构用于旧 run 的原地 resume；新实验使用独立 RUN_ID。

## 验证

运行单元和集成测试：

```bash
PYTHONPATH=. python tests/test_qwen3_vl_cross_view.py -v
```

测试覆盖 register 与本视角 patches 的双向梯度、视角内隔离、register 无绝对位置和 RoPE、global 层完整旁路、只合并 patches、原始权重加载、保存/重载、跨样本隔离、混合分辨率、视角排列、HF generation、checkpointed backward、加速器 BF16 和 YAML。2026-11-05 的本地验证：17 项通过（含 GPU BF16）；PyTorch 2.10.0+cu128、transformers 4.57.6，使用小型随机模型验证架构，未启动完整 Qwen3-VL-2B 训练或完整模型 smoke。

以下命令保留此前 ppu-dev / starvla-pai-check 的完整模型验证入口；2026-11-02 的历史报告对应未加入 registers 的版本：

```bash
PYTHONPATH=. /opt/ac2/bin/python tests/test_qwen3_vl_cross_view.py -v
PYTHONPATH=. /opt/ac2/bin/python tests/smoke_qwenpi_cross_view.py --output playground/Reports/qwenpi_da3_cross_view_20261102/full_model_smoke.json
```

完整模型 smoke 使用真实 Qwen3-VL-2B 与 QwenPI 动作头、3/2 混合视角，检查训练 loss/backward、register 梯度与动作推理；不更新参数、不提交训练或评测任务。重新运行时请为 `--output` 指定新路径，保留历史报告。

DA3 参考：https://arxiv.org/html/2511.10647v1#S3.SS2
