# LIBERO Object：width-error夹爪解码初测

日期：2026-10-23；主机：ppu-dev；运行容器：starvla-pai-check。

## 结果

只运行新解码，未重跑旧基线；仅评测Object。五个历史0/50任务各测前10个官方初始状态，再加ketchup回归控制10次，共60次。

| 任务 | 原阈值解码，历史结果 | 新width-error解码 |
|---|---:|---:|
| alphabet soup | 0/50（0%） | 9/10（90%） |
| ketchup | 50/50（100%） | 6/10（60%） |
| tomato sauce | 0/50（0%） | 9/10（90%） |
| milk | 0/50（0%） | 10/10（100%） |
| chocolate pudding | 0/50（0%） | 10/10（100%） |
| orange juice | 0/50（0%） | 10/10（100%） |

五个原全失败任务合计：历史0/250，新48/50（96%）。ketchup从历史50/50变为本次6/10（60%）。本次只测了有目的选择的6个子任务，不能据此计算或宣称完整Object套件成功率。

结论：新的宽度误差解码明显恢复了原全失败任务，但带来ketchup回归。因此它支持“原固定阈值是重要问题”的判断，尚不能作为无回归的通用修复。保留为可选模式，未替换默认规则。

## 修改内容

- umi16_to_osc新增gripper_mode=width_error：error=预测实测宽度−当前实测宽度；|error|≤1.5mm输出0；error>1.5mm输出-1（张开）；error<−1.5mm输出+1（闭合）。
- ModelClient每步接收当前两指间距，保留原位姿转换与16步chunk；直接返回gripper_command。
- 评测外层直接传递该-1/0/+1命令，避免原二值化将0变成闭合；legacy模型仍沿用原open_gripper处理。
- eval_libero.sh新增可选GRIPPER_MODE、GRIPPER_TOLERANCE环境变量，默认仍为threshold和0.0015。

启用新模式：
```bash
GRIPPER_MODE=width_error GRIPPER_TOLERANCE=0.0015 bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
```
其余CKPT、LIBERO_HOME、Python与端口参数仍按原评测配置提供。上面的普通入口会按原配置评测套件；本次缩小任务范围由保存的独立evaluate_ab.py执行。

## 权重、输入与协议

- checkpoint：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Checkpoints/multiview_libero_umi110k_gb256_backup/checkpoints/steps_15000_pytorch_model.pt
- 保留其config.yaml和dataset_statistics.json；未重训、未改权重、未替换归一化统计量。大小、mtime与服务启动记录一致，未计算8GB权重全量SHA256。
- 模型源码DualStreamQwenPI、LayerwiseFM_ActionHeader和policy_norm_processor的哈希，与此前固定图像诊断一致。旧模型两处norm默认none，未开启新LayerNorm。
- 模型输入仍为agentview、wrist两视角RGB及原始任务文字；256像素观测旋转180度后PIL bilinear缩放224。没有向模型增加state。当前EEF位姿和夹爪宽度只用于执行适配。
- 仍使用原评测的每步相机观测；未启用渲染优化。视频只保存每个chunk开始帧（双视角拼接，1.25fps），不是逐步录像。
- 每任务官方初态trial0..9，仿真seed7，先等待10步，最多280个策略执行步，动作chunk16，成功判据仍为原LIBERO done信号。
- 推理服务在每个chunk按固定公式重置采样seed，见protocol.json。这与原历史服务器连续随机数流不同，因此不是相同噪声的成对A/B。用户已要求只运行新解码。
- 两个PPU推理副本、8个CPU仿真worker；所有60次正常结束，无自动替换失败episode、无追加挑选种子。

## 环境与验证范围

- 为遵守仓库验证规则，仿真也在starvla-pai-check中运行，依赖隔离安装到本实验runtime_site。robosuite1.4.0、MuJoCo3.2.3与旧环境一致；Python3.12、NumPy1.26.4与旧libero-server的Python3.10、NumPy1.24.4不同。
- 容器原numba/LLVM JIT在构造环境时发生段错误；本次仿真统一设置NUMBA_DISABLE_JIT=1，未修改仿真源码。此环境差异也意味着不能把与历史基线的差异当作只改变一项的严格因果消融。
- 曾尝试按需渲染，但与缓存相机观测有少量像素差异，故未采用；正式60次全部使用原每步相机路径。
- 接口测试通过：闭合/保持/张开、缺失反馈拒绝、client→外层eval保持0命令、legacy回退。
- 128组随机位姿/动作对照：默认threshold与修改前逐位一致；width_error仅改变夹爪维，前6维位姿命令逐位一致。
- 同一官方初态下，两次固定命令仿真预检的状态和策略图像逐位一致；仅覆盖这个预检场景，不能推广为新旧运行环境逐位相同。
- 修改的Python均在指定容器编译通过；shell入口bash -n通过。
- 60条唯一结果、60份轨迹、60份chunk预测、60段视频均已核对完整性；见final_audit.json。

## 回归与边界

ketchup新结果6/10，不应隐藏。抽查其trial5、trial6的视频，抓取/搬运没有持续稳定保持；当前宽度误差规则会在预测波动超过容差时张开，而0只保持内部目标，不保证达到原持续闭合命令的夹持力。短暂误开与保持力度不足都是待验证原因，本次没有进一步调容差或增加锁存规则。

本次每任务仅10次，是初测。既未测其它Object任务，也未测Goal/Spatial/Long；不能将48/50推广为完整套件性能。原来的几乎全失败分布有了明显改善，但最终修复需要兼顾成功任务的回归。

## 保存位置

- summary.json：六任务新结果和历史计数。
- protocol.json：任务、初态、种子公式和执行参数。
- completion.json：8个worker退出码，均为0。
- final_audit.json：结果完整性、汇总、权重身份与工件SHA256。
- results/：逐次成功/失败、步数、耗时与命令计数。
- traces/：逐步数组，列为step、当前宽度、预测宽度、7维OSC命令、当前EEF位置。
- chunks/：每段完整16×16物理预测、chunk anchor和采样seed。
- videos/：双相机chunk采样录像。
- success_contact_sheet.png、failure_contact_sheet.png：已抽查的成功/失败视频拼图。
- before/、source_changes.json：改动前文件与哈希。

实验根目录：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Evaluation/libero_gripper_width_error_ab_20261023
