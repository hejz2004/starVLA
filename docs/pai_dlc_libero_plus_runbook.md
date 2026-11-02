# PAI DLC LIBERO-plus 评测操作记录

更新：2026-10-31。用于后续更换模型 checkpoint、重复提交同一协议的 LIBERO-plus 评测。代码、配置、日志与结果都保存在 `ppu-dev` 的远程仓库；本地入口不保存代码副本。

已跑通配置：**单节点 16 PPU、160 CPU、1600Gi 内存，每卡 20 worker，共 320 worker**。流程保持简单：`launch.json → launch.sh → 安装额外依赖 → editable 安装 starVLA → run_cluster.sh → 评测`。

## 1. 固定入口和资源

以下操作在 `ssh ppu-dev` 登录后的 Bash 中执行。不要把 `dev-h20` 的环境与此流程混用。

```bash
export REPO_ROOT=/mnt/workspace/Native_Policy/user/hjz/starVLA
export EVAL_BUNDLE="$REPO_ROOT/examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel"
cd "$REPO_ROOT"
```

| 项目 | 已验证配置 |
| --- | --- |
| 提交工具 | `dlc-submit-01/dlc_cli.py` |
| 提交工具 Python | `dlc-submit-01/.venv/bin/python`，沿用已有 CredentialsURI 认证 |
| 地域 / workspace / quota | `cn-wulanchabu` / `310375` / `quota1obhtzfmre1` |
| 镜像 | `egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/training-xpu-pytorch:26.04.post1` |
| 运行 Python | `/opt/ac2/bin/python`，Python 3.12 |
| DLC 规格 | `Master` 1 Pod；GPU `16`、CPU `160`、Memory `1600Gi`、SharedMemory `32Gi` |
| 平台设置 | Priority `6`、Driver `2.1.0`、EnableRDMA `true`、RestartPolicy `Never`，最长 1080 分钟 |
| 新模型配置模板 | [launch.template.json](../examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/launch.template.json) |
| 完整安装入口 | `launch.sh`；`run_dlc.sh` 转发到它 |
| 已准备环境入口 | `run_cluster.sh`；`scripts/run_libero_plus_parallel.sh` 转发到它 |

**现有 `pai_parallel/launch.json` 是最后一次历史续跑配置，带 `--resume` 和旧 RUN_ROOT。新模型从 `launch.template.json` 生成独立 JSON，不照搬历史续跑配置。** 每次实际提交的准确配置和 JobId 由提交工具保存在该次 `dlc-submit-01/runs/<RUN_ID>/` 中。

保持模板里的 `DataSources`、`UserVpc`、资源池和驱动配置，与本次成功作业一致。两条 NAS 挂载为：

| NAS URI | 容器路径 | 权限 |
| --- | --- | --- |
| `nas://291s1jv2cdy6o4bfz6g-mke60.cn-wulanchabu.nas.aliyuncs.com/lpy` | `/mnt/workspace` | RW |
| `nas://291s1jv2cdy6o4bfz6g-mke60.cn-wulanchabu.nas.aliyuncs.com/public` | `/mnt/nas/public` | RO |

模板中保留已验证的 VPC / Switch / SecurityGroup 字段。换地域、资源池或机器时应重新核实挂载，不能只改镜像名称。

## 2. 新模型只需设置 checkpoint 和独立运行名称

```bash
export CKPT_PATH="$REPO_ROOT/playground/Checkpoints/YOUR_MODEL_RUN/checkpoints/steps_YOUR_STEP_pytorch_model.pt"
export RUN_ID="libero_plus_modeltag_stepXXXX_16ppu_w20_$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="$REPO_ROOT/playground/Evaluation/$RUN_ID"
export JOB_JSON="$REPO_ROOT/dlc-submit-01/jobs/$RUN_ID.json"
export RUN_DIR="$REPO_ROOT/dlc-submit-01/runs/$RUN_ID"
```

先替换模型目录、步数及 `modeltag_stepXXXX`。checkpoint 的 run 目录应保留 `config.yaml`、`config.full.yaml`、`dataset_statistics.json`，配置中引用的本地 VLM / processor 文件也应在 DLC 挂载内可见。运行使用离线模型文件。

当前适配接口为 `libero-anchored-osc-goal-binary-v1`：action dimension 16、horizon 16、无策略 state 输入，再解码为 OSC7 和二值夹爪。相同接口的新 checkpoint 可复用流程；动作接口改变时先适配、验证，再用新目录评测。

生成配置，不修改历史任务或历史结果：

```bash
python3 - <<'PYCFG'
import json, os
from pathlib import Path
bundle = Path(os.environ['EVAL_BUNDLE'])
checkpoint = Path(os.environ['CKPT_PATH'])
assert checkpoint.is_file(), checkpoint
job = json.loads((bundle / 'launch.template.json').read_text())
job['DisplayName'] = 'lipeiyan-hjz-' + os.environ['RUN_ID'].replace('_', '-')
job['Envs'].update(
    RUN_ID=os.environ['RUN_ID'], RUN_ROOT=os.environ['RUN_ROOT'],
    CKPT_PATH=str(checkpoint), BOOTSTRAP_LOG=os.environ['RUN_ROOT'] + '.bootstrap.log')
path = Path(os.environ['JOB_JSON'])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as handle:
    json.dump(job, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(path)
PYCFG
```

新 checkpoint 必须使用新的 `RUN_ID`、`RUN_ROOT` 和回执目录；不加 `--resume`。不要预先创建 `RUN_DIR`，提交工具会创建它；已有回执目录不能删除后重新提交。

## 3. 安装依赖的职责

依赖文件位于评测入口目录：

- `requirements.txt` 引用仓库 `requirements-ppu.txt` 和 `requirements-simulation.txt`，另固定兼容版本的 headless OpenCV、h5py、albucore。
- 仿真版本包含 NumPy 1.26.4、MuJoCo 3.2.3、robosuite 1.4.0、Gym 0.25.2、Wand 0.6.13、`future==1.0.0`。
- `launch.sh` 安装 `libosmesa6 libgl1 libmagickwand-dev ffmpeg`，然后 `pip install -r .../requirements.txt`，再 `pip install --no-deps --no-build-isolation -e "$STARVLA_DIR"`。
- 安装时约束镜像自带的 PPU Torch、torchvision、torchaudio、triton、transformers、accelerate、deepspeed，避免被公共 PyPI 版本替换。

DLC 使用镜像内安装好的 site-packages，不依赖 DSW 旧的 `libero_plus_sim_runtime` overlay。那个 overlay 只用于既有开发容器的测试。

默认共享资源：

- Plus 源码：`playground/Artifacts/libero_plus_shared/source`，固定上游 commit `4976dc3`。
- 完整公共资产：`/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets`。上层 `READY.json` 标记 complete；评测仅读取它。
- 不再使用旧的不完整资产目录，也不默认混入另一份标准 LIBERO assets。

## 4. 提交前做有针对性的验证

仅换同接口 checkpoint 时，先在已准备好的 `starvla-pai-check` 容器做小规模真实评测。该容器必须能读取公共资产路径；Docker 命令使用绝对路径。

```bash
/etc/dsw/runtime/export_bin/docker exec \
  -e STARVLA_PYTHON=/opt/ac2/bin/python -e LIBERO_PYTHON=/opt/ac2/bin/python \
  -e LIBERO_HOME="$REPO_ROOT/playground/Artifacts/libero_plus_shared/source" \
  -e LIBERO_ASSETS=/mnt/workspace/Data/LIBERO_PLUS_EVAL_4976dc3/assets \
  -e LIBERO_SITE_PACKAGES="$REPO_ROOT/playground/Artifacts/libero_plus_sim_runtime" \
  -e CKPT_PATH="$CKPT_PATH" -e RUN_ROOT="${RUN_ROOT}_smoke" \
  -e EVAL_MODE=smoke -e NUM_CARDS=2 -e CPU_BUDGET=20 -e WORKERS_PER_CARD=20 \
  -e BASE_PORT=17883 \
  starvla-pai-check bash "$EVAL_BUNDLE/run_cluster.sh"
```

容器停止时先 `docker start starvla-pai-check`（使用上述 Docker 绝对路径）。若已有本地评测占用端口，另选 BASE_PORT。所有模型代码测试和 `py_compile` 均在该容器内执行。

**变更镜像、依赖或安装脚本时，用原始镜像的全新 Docker 再验证一次 bootstrap**，不要仅靠旧容器里已装好的包判断成功。本次采用以下挂载与设备方式，测试容器只有本机两张 PPU：

```bash
TEST_CONTAINER="starvla-plus-${RUN_ID}-fresh"
IMAGE='egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/training-xpu-pytorch:26.04.post1'
/etc/dsw/runtime/export_bin/docker run --name "$TEST_CONTAINER" \
  --network host --shm-size 32g \
  --device /dev/alixpu --device /dev/alixpu_ctl \
  --device /dev/alixpu_ppu0 --device /dev/alixpu_ppu1 \
  -v /mnt/nas/lpy:/mnt/workspace -v /mnt/nas/public:/mnt/nas/public:ro \
  -e CKPT_PATH="$CKPT_PATH" -e RUN_ROOT="${RUN_ROOT}_fresh_smoke" \
  -e EVAL_MODE=smoke -e NUM_CARDS=2 -e CPU_BUDGET=20 -e WORKERS_PER_CARD=20 \
  -e BASE_PORT=17883 \
  "$IMAGE" bash "$EVAL_BUNDLE/launch.sh"
/etc/dsw/runtime/export_bin/docker inspect "$TEST_CONTAINER" --format '{{.State.ExitCode}}'
```

默认 smoke 固定为四套各 task ID 0–4，共 20 回合；这批都是 Background 变体，并不代表全量成功率。改变样本集合后，成功数不能直接与旧 smoke 比较。视频逻辑改动时，可在启动命令末尾追加 `--suites libero_10 --task-ids 1268`，复查本次曾失败的长语言指令任务。正式评测仍使用完整 10030 回合。

验证完成看退出码、smoke 的 `completion.json`，并确认结果、视频与轨迹实际写出；不以进程启动或某个成功率阈值作为安装验证的替代。

## 5. 预览配置并提交 DLC

```bash
DLC_PY="$REPO_ROOT/dlc-submit-01/.venv/bin/python"
DLC_CLI="$REPO_ROOT/dlc-submit-01/dlc_cli.py"
"$DLC_PY" "$DLC_CLI" submit \
  --job "$JOB_JSON" --run-dir "$RUN_DIR" --work-root "$REPO_ROOT" \
  --region cn-wulanchabu --dry-run
```

预览确认单节点 16 PPU、160 CPU、1600Gi、每卡 20 worker；挂载与模板一致；checkpoint 和输出目录是本次模型的路径。然后去掉 dry-run 实际提交一次：

```bash
"$DLC_PY" "$DLC_CLI" submit \
  --job "$JOB_JSON" --run-dir "$RUN_DIR" --work-root "$REPO_ROOT" \
  --region cn-wulanchabu
```

回执包括 `job.json`、`identity.json`、`submission-attempt.json`、`submission-response.json`、`submission.json`。若提交返回不明，先用同一回执目录执行 `list` 核对是否已创建，不能盲目重发。

## 6. 查看启动和进度

```bash
"$DLC_PY" "$DLC_CLI" status --run-dir "$RUN_DIR" --work-root "$REPO_ROOT"
"$DLC_PY" "$DLC_CLI" logs --run-dir "$RUN_DIR" --work-root "$REPO_ROOT"
tail -n 30 "${RUN_ROOT}.bootstrap.log"
cat "$RUN_ROOT/summary.json"
```

`Running` 只表示平台进程在运行，也可能仍在安装依赖或加载模型。正常启动的证据依次是：安装 starVLA 完成、16 个 policy ready、运行目录 `launch.json` 记录 320 workers、`summary.json` 的 completed 持续增加且 error 为空。模型按每批最多 4 份加载，冷启动需要数分钟。

| 路径 | 用途 |
| --- | --- |
| `${RUN_ROOT}.bootstrap.log` | 安装依赖及控制器 stdout；配置 BOOTSTRAP_LOG 时以该值为准 |
| `${RUN_ROOT}/bootstrap.log` | 控制器运行日志 |
| `preflight/inputs.log`、`preflight/simulation.log` | 资源及仿真初始化问题 |
| `policies/device_*/policy.log` | 模型加载、连接、推理日志 |
| `workers/d*/run.log`、`workers/d*/progress.json` | 每个 worker 日志、当前任务和步数心跳 |
| `errors/*.json` | 运行异常；任务正常失败是结果 success=false，不属于这里的错误 |
| `summary.json`、`<Suite>/summary.json` | 总体、套件和七类扰动统计 |
| `completion.json` | 所有计划回合完成的记录 |

完整完成同时核对 DLC `Succeeded`、`completion.json` complete、`summary.json` 为 10030/10030。正式结果目录包含 `Spatial`、`Object`、`Goal`、`Long` 四个子目录。

视频路径为 `<Suite>/videos/<episode_key>/<perturbation>_suc.mp4` 或 `_fail.mp4`。任务目录保证不重名；文件名不含完整指令。顶部信息栏显示套件、任务编号、扰动和实际语言指令，下面是主摄左、腕摄右的双视角。信息栏只写入视频，不进入模型输入。结果 JSON 在 `results/`，OSC7 动作轨迹在 `traces/`。

## 7. 中断后续跑与更换模型的区别

**同一 checkpoint、同一评测协议的中断任务**：新建提交身份和新的回执目录，保留旧 RUN_ROOT，在 UserCommand 的 `launch.sh` 后加 `--resume`；BOOTSTRAP_LOG 换一个新名字，方便区分尝试。可以调整 worker 数和 CPU 分配。控制器读取已完整提交的结果，成功和正常失败的回合都保留，只运行尚未完成的回合。

**更换 checkpoint、动作/输入协议、任务集合或改变会影响成绩的代码**：使用全新输出目录，不能把旧成绩当作新模型的已完成结果，也不能通过修改协议记录强行续跑。

本次曾对“视频保存与 worker 默认值”做一次明确归档的兼容迁移：先保存旧协议和结果 JSON，再调整视频路径与协议记录，保留轨迹和任务成败。迁移代码与记录在 `playground/Reports/libero_plus_video_w20_20261030/`，包含固定的历史数量和哈希，**不能当通用续跑脚本反复执行**。以后的普通续跑不需要此操作；若修改代码触发协议不匹配，先确认差异，不关闭检查或无条件改摘要。

## 8. 本次遇到的问题与已落实的处理

| 问题 | 处理 |
| --- | --- |
| Ubuntu 24.04 中旧 ImageMagick 包名不可用 | 使用已验证的 `libmagickwand-dev`，连同 OSMesa 和 ffmpeg 显式安装 |
| BDDL 导入报缺少 future | 在 requirements 中显式固定 `future==1.0.0` |
| 旧资产目录不完整 | 使用 WYT 完整公共 `LIBERO_PLUS_EVAL_4976dc3/assets` |
| 语言扰动指令过长导致视频文件名超过限制 | 按任务目录保存，视频名仅含扰动类别和 suc/fail |
| 同版本 Gym 重建后 METADATA 哈希不同 | 比较实际安装版本；不对会重排的 Requires-Dist 文本做整文件哈希 |
| thop 的目录名版本与真实 Version 字段格式不同 | 从 `importlib.metadata` 读取真实版本，不从 dist-info 文件夹名猜版本 |
| 把 Running 误当评测已正常进行 | 检查 policy ready、worker 计划、回合进度和错误日志 |
| 不同 smoke 成功数看起来变化很大 | 固定并记录任务集合；报告清楚功能验证与全量成绩的区别 |

## 9. 固定评测口径与汇报方式

四套共 10030 个变体，每个变体使用 initial state 0、trial 0 跑一次。仿真 seed=7、等待 10 步；步数上限 Spatial 220、Object 280、Goal 300、Long 520。每卡一份 BF16 模型，CPU OSMesa 渲染，线程受限，320 worker 共享 160 CPU。全局队列分配任务，不使用 DDP 或跨 worker 推理 batch。

指令协议为 `libero-plus-bddl-v1`：沿用环境规则，定位 BDDL 时去掉 `_view_` 及后续运行时后缀；普通扰动使用该 BDDL 的 `language_instruction`，语言扰动保留 `_language_` 变体标识并使用对应的改写指令。仅整理空白，不把文件名中的扰动编号输入模型。结果记录实际指令、来源 BDDL 和原始任务名。历史 2026-10-30 结果仍属于原样 `task.language` 协议，新评测使用新输出目录。主摄/腕摄 RGB 各旋转 180°，按主摄在前、腕摄在后，用 PIL BICUBIC 缩放到 224×224。策略输入只有 image 和 lang。策略推理 RNG 未显式固定，不能承诺重复运行逐动作一致；不能把 worker 数变化后的随机采样差异当作确定性复现。

从 `summary.json` 的 suites/categories 汇报下列列序：

| 汇报列 | 记录中的 category |
| --- | --- |
| Camera | Camera Viewpoints |
| Robot | Robot Initial States |
| Language | Language Instructions |
| Light | Light Conditions |
| Background | Background Textures |
| Noise | Sensor Noise |
| Layout | Objects Layout |
| Total | 所有回合 |

按 Spatial、Object、Goal、Long 分行，并给出整体行及成功数/总数。整体成功率为汇总成功数除以汇总回合数，不是简单平均各套件百分比。

## 10. 首次完整运行基准

- Checkpoint：`playground/Checkpoints/multiview-umi110k-libero-posttrain/checkpoints/steps_15000_pytorch_model.pt`。
- 成功作业：`dlcr5m03vajqiiuo`，2026-10-30 北京时间 13:25 完成，平台 Succeeded。
- 提交回执：`dlc-submit-01/runs/libero_plus_15000_16ppu_w20_resume_20261030_110442/`。
- 结果目录：`playground/Evaluation/libero_plus_15000_16ppu_w10_20261030_092954/`。目录名保留历史 w10；最后一次运行实际为每卡 20 worker。
- 全量成绩：4580/10030，45.66%；Spatial 1343/2402、Object 1184/2518、Goal 1042/2591、Long 1011/2519。
- 最后一次续跑约 2 小时 17 分钟，包含安装与加载，并复用了之前完成的 1232 回合；这不是从零跑完 10030 回合的独立耗时基准。
- 前 1232 个视频保留原画面、仅调整文件名；之后的视频包含顶部信息栏。
- 环境安装记录：`playground/Reports/libero_plus_dlc_bootstrap_20261030/`。
- 视频、20-worker 验证及迁移记录：`playground/Reports/libero_plus_video_w20_20261030/`。

详细调度与文件格式见[评测入口 README](../examples/simBenchmarks/LIBERO-plus/eval_files/pai_parallel/README.md)；提交工具字段说明见[现有 DLC CLI 手册](../dlc-submit-01/README.md)。2026-10-21 的旧迁移文档仅作历史记录。
