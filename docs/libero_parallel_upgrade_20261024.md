# LIBERO guarded接口与双PPU后台评测交接

2026-10-24，ppu-dev。用户要求停止额外测试并后台开跑。

- 常规eval_libero.py/.sh默认采用guarded_width_error；原threshold、width_error、intent_latch显式选项保留。低层ModelClient/umi16_to_osc默认仍为threshold，兼容未提供宽度反馈的旧调用；原生7维路径不变。
- 新入口：scripts/run_libero_eval_2ppu.sh；通用入口与说明：examples/simBenchmarks/LIBERO/eval_files/eval_libero_parallel.sh、README_parallel.md。
- 默认每卡一份模型、每卡两个独立仿真进程，逐episode/chunk固定采样种子，原子保存结果和轨迹，支持筛选、断点续跑和汇总。输出目录锁阻止重复写入；进程退出清理自己的子进程。
- 已完成兼容性/分片检查及两条真实双卡Ketchup流程检查，两条成功；不是完整回归结果，采样协议也不同于前一天的试验。用户要求后不再追加测试。

## 当前后台任务

- 模型：playground/Checkpoints/multiview_libero_umi110k_gb256_backup/checkpoints/steps_15000_pytorch_model.pt
- 范围：仅新接口，Object十任务各50次，共500条，不重跑旧接口。
- 两张PPU，四个仿真worker，端口11093/11094。
- 输出：playground/Evaluation/libero_object_guarded_parallel_20261024/
- 主日志：playground/Reports/libero_parallel_upgrade_20261024/object_background.log
- 启动记录：同报告目录background_run.json；容器PID见object_background.pid。
- 使用docker exec -d无TTY启动，启动器独立会话，不依赖SSH或本地终端。关闭SSH后已从新的SSH连接确认PID存活、TTY为?，protocol.json/summary.json已写入，期望500条。
- 查看summary.json了解进度；全部完成写completion.json。失败写错误到summary.json/主日志，保留已提交结果。

## 如需续跑

在ppu-dev仓库根目录，确保无同目录启动器运行：

```bash
OUTPUT_DIR="$PWD/playground/Evaluation/libero_object_guarded_parallel_20261024" \
  bash scripts/run_libero_eval_2ppu.sh --resume
```

上面是手动前台续跑示例；本次实际已通过Docker detached启动，无需再运行它。模型、协议或受追踪Python源码变化时拒绝混用旧结果。默认新模式单卡视频目录已加模式后缀，避免覆盖旧threshold视频。

## 范围更正

用户要求Object、Long（libero_10）与Goal，共1500条。原Object任务尚未完成任何episode，已正常停止原启动器及其子进程，保留旧目录。新任务仍以docker exec -d启动，两卡四worker。

当前结果目录：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Evaluation/libero_object_goal_long_guarded_parallel_20261024

当前主日志：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Reports/libero_parallel_upgrade_20261024/three_suites_background.log

当前启动记录：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Reports/libero_parallel_upgrade_20261024/background_run.json

续跑时使用新OUTPUT_DIR，并保留 --suites libero_object libero_10 libero_goal --num-trials-per-task 50 --resume。
