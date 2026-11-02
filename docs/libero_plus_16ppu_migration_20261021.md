> 2026-10-30 更新：当前已完成单节点16 PPU / 160 CPU / 1600Gi、每卡20 worker的完整DLC评测。后续操作以 [PAI DLC LIBERO-plus 操作记录](pai_dlc_libero_plus_runbook.md) 为准。下文是2026-10-21旧UMI16/8-worker方案的历史记录。

# LIBERO-plus 16 卡并行评测迁移

实现位于 `examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/`，通用入口为 `scripts/run_libero_plus_parallel.sh`。直接使用现有镜像和共享文件；不构建镜像、不安装依赖、不复制资产、不提交集群任务。旧名称 `run_libero_plus_16ppu.sh` 和 `run_dlc.sh` 保留为同一入口的别名。

## 运行

在已有环境中设置两个解释器和共享资源路径：

```bash
STARVLA_PYTHON=/path/to/policy/python \
LIBERO_PYTHON=/path/to/libero/python \
LIBERO_HOME=/shared/LIBERO-plus \
LIBERO_ASSETS=/shared/libero_plus_assets \
bash scripts/run_libero_plus_parallel.sh
```

两个解释器也可以相同，默认均为 `python`。代码通过标准 PyTorch CUDA 接口工作，没有新增 PPU 专属依赖；后续 H100 使用对应现有环境即可。本次验证在 PPU 上完成，H100 的镜像/驱动/依赖仍需在目标节点运行前置检查。

默认使用当前仓库的：
`playground/Checkpoints/multiview_libero_umi110k_gb256_backup/checkpoints/steps_15000_pytorch_model.pt`。
可用 `CKPT_PATH` 指定该权重在共享盘上的绝对路径；本入口针对它的 DualStreamQwenPI UMI16/chunk16/flow4/bf16/双摄/无策略state协议进行校验。配套config与dataset_statistics必须仍位于checkpoint所在run目录。

模型图像采用原普通LIBERO客户端的PIL bilinear 224缩放和主摄/腕摄顺序；UMI16动作通过现有客户端转换成OSC。EEF位姿只用于转换，不发送为策略state。每worker独立保存chunk锚点和缓存。若VLM目录重定位，可设置 `VLM_DIR`，仅覆盖模型配置的文件路径。

源代码部分会通过 `__file__` 定位assets。入口会在本次输出目录创建浅层源码与资产符号链接视图，并生成 `libero_config/config.yaml`；不会复制源码数据或资产，不会修改共享目录中的原始链接或全局LIBERO配置。如果Plus资产仅含扩展部分，设置 `LIBERO_BASE_ASSETS=/shared/standard_LIBERO/libero/libero/assets` 补齐基础目录（如articulated_objects）；同名顶层目录以Plus为准。镜像中需要已有OSMesa/ImageMagick及匹配的仿真依赖；协议检查使用MuJoCo3.2.3、robosuite1.4.0。

## 两卡试跑

```bash
EVAL_MODE=smoke NUM_CARDS=2 \
STARVLA_PYTHON=/path/to/policy/python \
LIBERO_PYTHON=/path/to/libero/python \
LIBERO_HOME=/shared/LIBERO-plus \
LIBERO_ASSETS=/shared/libero_plus_assets \
bash scripts/run_libero_plus_parallel.sh
```

此模式期望恰好看见2张卡（如需要，可在启动前设 `CUDA_VISIBLE_DEVICES=0,1`）。每卡8个worker，分别跑goal/spatial前8个任务，共16回合。真实视频解码和完整性审计通过后才接受。试跑不进入完整评测，也不计正式成绩。

## 正式16卡

默认 `EVAL_MODE=full NUM_CARDS=16`。须为单节点16卡。入口依次执行：共享配置检查 → 仿真导入与suite数量检查 → 四套场景/五类噪声真实双摄渲染 → 各卡bf16实算 → 单卡单回合 → 两卡16回合 → 正式16卡128worker共10030回合 → 汇总和产物审计。

卡分配沿用wyt成功实现：0–6跑libero_10，7–9跑goal，10–12跑object，13–15跑spatial；默认端口9883–9898。每卡一份完整模型、8个CPU仿真进程。卡间独立运行，卡内WebSocket请求逐个推理，不合并batch，不使用DDP。

每变体只使用第0个初态评测1次；worker NumPy和环境seed=7；策略不显式设置RNG种子。稳定等待10步，上限spatial220/object280/goal300/10为520。

## 输出与完成判据

输出默认：`playground/Evaluation/libero_plus_15k_<时间>/`。可设置唯一 `RUN_ID` 或绝对 `RUN_ROOT`，禁止复用已有目录。`BASE_PORT`可指定，`SERVER_READY_TIMEOUT`默认900秒。

- `bootstrap.log`、`preflight/*.log`：启动与前置检查。
- `gates/single_trial`、`gates/smoke_2card`：正式模式的试跑。
- `formal/manifest.tsv`：16卡任务区间与端口。
- `formal/<suite>/card_*/policy_server.log`：模型日志。
- `worker_*/run.log`、`logs/trials.jsonl`、`videos`、`actions`：逐worker及逐回合记录。
- `formal/summary.json`、`summary.csv`：按回合计数汇总。
- `formal/COMPLETION_AUDIT.json`及根目录`PIPELINE_COMPLETED.json`：完整完成的证明。

正式完成需要16卡、128worker、10030个互不重复的回合、非空视频和有限动作轨迹全部核对通过。控制器失败即停，不自动重跑；先查看FAILED/STATE及日志，再用新RUN_ID尝试。

`UPSTREAM.json`记录wyt成功运行所用上游代码的路径和SHA256。原普通LIBERO入口、训练代码和wyt/zth的文件均未修改。验证记录见 `docs/libero_plus_16ppu_migration_20261021_validation.md`。
