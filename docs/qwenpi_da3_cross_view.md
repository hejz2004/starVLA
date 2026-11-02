# QwenPI 后段交替多视角注意力

本分支 `experiment/cross-view-architecture` 从原始 QwenPI 实现出发，移除了 DualStreamQwenPI、QWen3DualStream、旧 vendored 双流模型及专用调试测试。历史报告和 checkpoint 保留，不迁移旧权重。

## 实现

入口是 `framework.name: QwenPI`，动作头和 VLM 语言部分沿用 QwenPI。QWen3.py 选择新的 `qwen3_vl_cross_view` 模型；它直接基于 transformers 4.57.6 原始 Qwen3-VL 建立，保留参数名与数量。

```yaml
framework:
  name: QwenPI
  qwenvl:
    load_weights: true
    cross_view_attention:
      enabled: true
      start_layer: 16
```

层编号从 0 开始。24 层中 0–15 为视角内注意力，16/18/20/22 为跨视角注意力，17/19/21/23 为视角内注意力。`start_layer: null` 按模型深度取 floor(2*depth/3)。关闭 enabled 可作为原始视觉注意力对照。

本实现的后段从 global 开始交替，这是本实验明确选择的顺序。借鉴 DA3 的后段交替范围与全局位置处理，并非完整复制其几何网络、camera token、DPT 或几何监督。

跨视角层直接更新 hidden_states，融合后的特征继续进入后续视觉层，并由原有 DeepStack / final merger 输出。没有 shared token、gate、独立交互投影或额外视觉 block。原有 DeepStack 的第 5/11 层输出仍为单视角特征，第 17 层输出及最终 merger 才包含视觉跨视角融合。

packed 图像通过 input_ids 中的 vision_start/image token 推导每个样本的图像数量；global attention 按样本切分，local 按图像切分，支持混合视角数量和分辨率。文本样本可以没有图像。直接调用视觉塔需传 scene_image_counts，缺少边界会报错，不能把 batch 默认为一个场景。此实验仅支持静态图像视角；启用后的视频输入明确拒绝。

local 使用原始二维 RoPE。global 使用 identity RoPE（无二维相对旋转），不把不同相机像素坐标当成公共坐标；原始绝对位置嵌入仍保留。不新增相机 ID 或外参。本实现单视角也可运行，但启用模式中的 global 位置处理与原始模型不同；精确原始对照需 enabled=false。

UMI/LIBERO 的 action_mask 支持被保留到 QwenPI，含 diffusion repeat 对齐，避免把 LIBERO 的未使用右臂维度加入损失。支持 VLM gradient checkpointing。

## 配置与启动

- UMI：`examples/umi/train_files/starvla_qwenpi_da3_umi_v2.yaml`
- UMI 冒烟：`examples/umi/train_files/starvla_qwenpi_da3_umi_smoke.yaml`
- LIBERO controller-goal：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero_action_goals.yaml`
- LIBERO 原有 relative-EEF 变体：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero.yaml`
- LIBERO 冒烟：`examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_da3_libero_smoke.yaml`

现有 UMI / LIBERO 启动脚本默认路径已更新。训练数据、动作定义、原有 state 设置与训练超参数保持原 YAML 选择：UMI state=true，LIBERO state=false。做统一无状态实验需显式把 UMI use_state 设为 false。UMI 默认读原始 VLM 预训练参数；完全随机初始化可覆盖 `framework.qwenvl.load_weights=false`。

所有 LIBERO 配置的旧 UMI checkpoint 默认路径已清空；正式配置默认从原始 VLM 初始化。新架构 UMI 训练完成后，通过 `PRETRAINED_CHECKPOINT=/path/to/new/checkpoint.pt` 调用 `run_libero_train.sh`。迁移时需使 action head 的 state 配置与目标实验一致。

包含 shared_init / interaction_stages 的旧双流 checkpoint 会明确报错，包括 strict=false 加载；旧 DualStreamQwenPI 配置不再受支持。QwenPI 原始权重的参数结构兼容，但训练策略与注意力范围必须记录，不代表可以直接把旧训练结果当作新架构实验。

## 验证

在 ppu-dev 的 starvla-pai-check 容器运行：

```bash
PYTHONPATH=. /opt/ac2/bin/python tests/test_qwen3_vl_cross_view.py -v
PYTHONPATH=. /opt/ac2/bin/python tests/smoke_qwenpi_cross_view.py --output playground/Reports/qwenpi_da3_cross_view_20261102/full_model_smoke.json
```

测试覆盖原始输出等价（disabled）、跨视角梯度、跨样本隔离、混合分辨率、融合传播到后续 local 层、位置处理、序列化、HF generation、checkpointed backward、加速器 BF16 和 YAML。完整模型 smoke 使用真实 Qwen3-VL-2B 与 QwenPI 动作头、3/2 混合视角，检查训练 loss/backward 与动作推理；不更新参数、不提交训练或评测任务。

DA3 参考：https://arxiv.org/html/2511.10647v1#S3.SS2
