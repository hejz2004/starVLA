# LIBERO / LIBERO-plus：独立 16×A100 评测任务

2026-10-31 已按用户要求依次提交：LIBERO `dlc56saa3xeaxtci`、LIBERO-plus `dlc6upukrli84awz`。提交后均进入 A100 队列；无自动重提进程。

- 入口：`launch.sh`，加载已成功使用的 A100 镜像配套 NAS 环境，不安装或重建 CUDA 依赖。
- 调度：`run_node.py`。每个作业 Master + Worker 各 8 GPU、80 CPU、800Gi 内存、32Gi shm；每卡 20 个仿真 worker。
- 节点 0：Spatial、Goal；节点 1：Object、Long。套件互不重叠。
- LIBERO：每套件 10 任务 × 50 初态，合计 2000 回合；使用 `/opt/qwenpi/libero` 仿真环境。
- LIBERO-plus：四套件全量 10030 个变体，每变体一次；使用 `/opt/qwenpi/plus`，保留 `libero-plus-bddl-v1` 指令协议。
- 策略均使用 `/opt/qwenpi/train`、BF16、最新 anchored-goal binary 接口。保留原有视频和动作轨迹。
- 权重：本次继承 VLM 的 UMI98k→LIBERO 训练的 `checkpoints/steps_7500_pytorch_model.pt`，需训练完成并保存后才能启动。
- 镜像：`dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/cloud-dsw-user/users:u1887578208171147-1083222-20260926181042`。
- 资源：cn-wulanchabu / workspace 302261 / quota quotan1g83uwyake；NAS、网络、驱动沿用当前 A100 成功训练配置。

任务文件位于仓库根目录下：

```
dlc-submit-01/jobs/libero_vlminit_umi98k_step7500_16a100_20261031.json
dlc-submit-01/jobs/libero_plus_vlminit_umi98k_step7500_16a100_20261031.json
```

`prepared_jobs.json` 记录两份配置及对应回执/输出路径，仅为清单，不会触发提交。JSON 中保留用户 W&B key，权限为 600；评测关闭 W&B，不打印完整配置。

本次提交命令如下（已执行；不要对同一回执目录重复提交）：

```bash
cd /mnt/workspace/Native_Policy/user/hjz/starVLA
for benchmark in libero libero_plus; do
  name="${benchmark}_vlminit_umi98k_step7500_16a100_20261031"
  dlc-submit-01/.venv/bin/python dlc-submit-01/dlc_cli.py submit \
    --job "dlc-submit-01/jobs/${name}.json" \
    --run-dir "dlc-submit-01/runs/${name}" \
    --work-root "$PWD" --region cn-wulanchabu
done
```

评测根目录含 `node0/`、`node1/` 的独立结果和协议；Spatial/Object/Goal/Long 链接至对应节点。主节点定期合并根目录 `summary.json`，两个节点完成且覆盖全部回合后才生成 `completion.json`。`launch/node*.log` 为节点启动及控制器日志。

验证范围：已在 A100 上检查两套仿真环境导入及官方套件数量；本地 PPU 容器只做脚本语法和任务覆盖/汇总逻辑检查。尚未运行这个两节点 A100 评测入口的真实 rollout，不能把脚本检查当作完整评测验证。

## 2026-11-01 原始 VLM 直接后训练实验

当前自动提交清单：`dlc-submit-01/jobs/current-vlm-direct-evaluations.json`；训练 `dlcykawazci93mtx` 成功保存最终权重后，由 `scripts/monitor_train_then_eval.py` 提交两个独立 16 A100 评测任务。历史清单 `prepared_jobs.json` 保留原任务信息。

此前两项 A100 评测因汇总读取 NAS 文件出现 ESTALE 中断；`run_node.py` 现对该错误推迟本轮汇总并下轮重试，其他错误仍正常报出。

## 2026-11-02 统一 MuJoCo 版本

PPU / A100 的标准 LIBERO 和 LIBERO-plus 均要求 MuJoCo **3.2.3**、robosuite **1.4.0**。用户确认修改后的动作数据在 MuJoCo 3.2.3 下生成。

NAS runtime 内的 `/opt/qwenpi/libero` 原为 MuJoCo 2.3.7；`launch.sh` 现在用仓库 `playground/Artifacts/mujoco_3.2.3_cp310/` 的离线 wheel 将每个作业容器内的该环境升级到 3.2.3（`--no-deps`，保留其他依赖）。不修改共享 NAS runtime 或 CUDA 环境。Plus 已为 3.2.3，无须升级。

标准 LIBERO 的控制器在加载策略前检查实际导入版本，单进程评测入口也检查；Plus 沿用已有检查。版本不符会退出。运行目录的 `simulation_runtime.json` 记录实际版本。

已完成的 A100 标准 LIBERO 2.3.7 结果属于旧环境，不能标记为 3.2.3；需要使用新输出目录重评，不续跑旧目录。本次仅统一后续启动配置，未提交重评任务。PPU 容器的 CPU 验证不代表 A100 GPU rollout 验证。
