# 旧 UMI checkpoint 在 LIBERO 后训练时新增 LayerNorm 的兼容性分析

日期：2026-10-21。基于当前随机初始化 UMI110k + LayerNorm 后训练及只读 CPU decoder 实验。

## 结论

旧 checkpoint 加载成功，不代表插入归一化后的模型保持原有预测函数。初始 loss 升高是可解释的接口分布变化；不能把所有高位波动都当成无关紧要，也不能在几十步内宣称长期不收敛。

此前只验证了新增无仿射参数 LayerNorm 不增加参数键、旧权重可以完整加载，这不足以保证预测行为兼容；这一点需要明确修正。

## 1. 当前曲线并非完全不降

截至第62个optimizer update，rank0日志口径为每次更新最后一个microbatch的loss：

| 区间 | loss均值 | 标准差 |
|---|---:|---:|
| 1–20 | 1.165674 | 0.021935 |
| 21–50 | 1.142273 | 0.030322 |
| 51–62 | 1.099569 | 0.027166 |

第62步loss为1.080101；rank1相应末段均值1.112944。存在短期下降，但距离旧实验的低loss仍有明显差距。此时尚处于750步warmup，不能保证最终能恢复旧模型表现。

## 2. 无参数 LayerNorm 仍改变模型函数

无仿射参数的LayerNorm对每个token执行 `(h - mean(h)) / sqrt(var(h) + eps)`。旧解码器计算 `decoder(h)`；新增模块后计算 `decoder(LN(h))`。相同的decoder权重在不同输入上会给出不同输出。

此外，cross_condition_norm改变了每层交叉注意力用于生成K/V的条件特征，所以DiT产生的h本身也会变化。当前同时改变两个接口，不能单独把影响归到其中一处。

PyTorch官方定义：https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html

## 3. 真实训练中测到的变化

随机初始化 UMI110k checkpoint + 两处LN，首次前向记录：

- 首个被观测cross-condition token向量RMS：0.700274 → 0.999990。
- 首个被观测动作decoder输入token向量RMS：21.368586 → 1.000000。
- 后者absmax：465.412048 → 21.763477。
- 两卡完整checkpoint加载均为missing_keys=[]、unexpected_keys=[]。

这些是首样本的选定向量，不是全层统计；也不是“原无归一化完整模型的decoder输入分布”，因为当前上游cross-condition LN已经开启。它们仍直接证明decoder边界确实对旧解码器的输入进行了大幅变换。

## 4. 同一 decoder 权重的 CPU 对照

从真实UMI110k checkpoint内存映射读取decoder的两层MLP权重；使用同一人工输入h，分别执行decoder(h)与decoder(LN(h))。输入维度及权重真实，h是按观测RMS构造的高斯向量，不是实际LIBERO隐藏态。

- 输入RMS：21.383450。
- 归一化后输入RMS：1.000000。
- 不加LN的decoder输出RMS：8.848960。
- 加LN的decoder输出RMS：0.414096。
- 两份输出RMSE：8.435597。

此对照证明“参数完全相同也可以因LN产生不同输出”，不用于估计实际模型loss、策略质量或归因具体退化程度。实验只使用CPU，没有改变正在训练的权重。

## 5. 为什么旧随机组尤其会失去低起点

旧随机组已通过UMI预训练联合调整VLM、DiT和decoder；其低LIBERO起始loss来自这套完整函数的迁移。只在后训练时插入LN，相当于改变已有模块间的输入约定。原参数保留，但模块必须重新适应新的条件/动作特征分布。

这与从UMI预训练第一步就使用LN不同。归一化改善新训练的数值条件，并不保证它能在一个已训练完成、没有LN的模型中无缝插入。

## 6. 当前对比的边界与建议

- 当前曲线与旧日志还存在调度差异：旧学习率推进约为16倍，当前已修复；当前warmup750，旧名义warmup5000、实际约312.5更新。16卡换2卡也改变了样本组合，尽管global batch都为256。
- 为复现旧checkpoint的迁移行为，应使用与其预训练一致的归一化配置，并在同一当前trainer、同一学习率设置下建立对照。
- 最有针对性的诊断是固定相同输入、时间和噪声，分别测两处LN都关、仅条件LN、仅decoder LN、两处都开；该完整模型对照本次尚未进行。
- 如果最终架构需要两处LN，预训练阶段开启、后训练保持相同配置是更一致的实验路线。已有无LN checkpoint也可能通过适配训练恢复，但需要验证，不能凭“没有新增参数”作保证。

当前随机组训练仍在后台运行，本次分析没有停训练或更改其配置。原始证据：`playground/libero_ln_recheck_20261021/decoder_ln_distribution_probe.json`，以及随机run的 `telemetry/layernorm_rank_0.jsonl`、`telemetry/rank_0.jsonl`。
