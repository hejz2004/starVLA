# LIBERO-plus 并行代码验证结果

验证日期：2026-10-21。完成状态：真实两卡链路跑通，未执行完整16卡/H100评测。

- 17项回归测试通过：分片与产物审计、失败阻断/防重跑、UMI16/chunk16调度与worker锚点隔离、共享资产链接视图、错误checkpoint拒绝。
- 最终Python编译和所有Shell入口语法检查通过。
- 指定15000步checkpoint实际PPU推理返回有限的[1,16,16]动作。
- 四个suite和五类噪声真实双摄渲染通过，共18张图像；2张PPU各自bf16矩阵运算通过。
- 2个模型服务、每卡8个真实CPU仿真worker，共16个回合全部退出0。
- 16个逐回合记录、16个非空视频、16条有限动作轨迹审计通过；全部16个视频已解码核对帧数和首尾帧。
- 测试为goal/spatial各8个Background Textures变体，任务成功数为0/16。这是运行链路验收，不能据此确认模型在Plus上的效果；模型效果仍需另行调查。

真实联调输出：`/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Evaluation/libero_plus_parallel_real_smoke_20261021_v4`。

关键证据：`smoke_2card/STAGE_ACCEPTED.json`、`COMPLETION_AUDIT.json`、`summary.json`，以及`preflight/cpu_render.json`、`ppu_compute.json`。

完整验证JSON与源码SHA256：`playground/libero_plus_parallel_checks_20261021/validation.json`。

依照用户授权，PPU模型留在原starvla-pai-check容器内，CPU仿真在临时容器中只读挂载共享目录进行。共享Plus资产缺少articulated_objects，通过普通LIBERO现存基础资产的符号链接补齐；不复制资产，不修改共享源码及原始资产链接。测试期间没有构建镜像或提交集群作业。

已停止本次两个模型进程及临时CPU验证容器，移除了临时运行环境归档、原容器内临时Python与编译缓存。保留日志、视频、动作及审计记录。

正式入口与使用方式见 `docs/libero_plus_16ppu_migration_20261021.md`。当前完整预设为单节点16卡；未来H100环境及完整16卡运行尚未实测。
