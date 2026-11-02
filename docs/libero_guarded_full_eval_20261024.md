# guarded_width_error：Object / Long / Goal完整评测结果

日期：2026-10-24；主机ppu-dev；检查点为 multiview_libero_umi110k_gb256_backup 的steps_15000；未重新训练。

## 结果

1500/1500条已完成，1185次成功，三套件平均79.0%。每套件十任务，每任务50个官方初始化状态。模式为guarded_width_error，容差1.5mm，阈值来自原模型统计（约41.932mm）。没有启用intent_latch或接触锁存。

| 套件 | 旧threshold | 本次guarded | 变化（百分点） |
|---|---:|---:|---:|
| libero_object | 166/500（33.2%） | 480/500（96.0%） | +62.8 |
| libero_10 | 224/500（44.8%） | 365/500（73.0%） | +28.2 |
| libero_goal | 311/500（62.2%） | 340/500（68.0%） | +5.8 |

三套件历史合计701/1500（46.73%），本次1185/1500（79.0%），增加484次成功、32.27个百分点。Spatial未在本轮重评，不能据此发布四套件新分数。

## 重点结论

- Object由两极分化变为全部任务88%及以上：此前六个≤2%的任务现为88%、100%、96%、96%、98%、100%。宽物体任务恢复与先前的夹爪语义诊断一致；历史对比不构成严格单因素因果实验。
- Ketchup48/50（96%），历史50/50（100%），减少2次成功。较此前width_error的小样本结果更接近历史水平，但采样协议与样本数不同，不能作严格成对比较，也不能宣称零回归。两条失败原因未逐视频定位。
- Long此前四个0%任务：alphabet soup+tomato sauce到70%；cream cheese+butter到80%；white mug+chocolate pudding到66%；alphabet soup+cream cheese到92%。
- Long仍有退化：开灶并放摩卡壶64%→54%；碗入下抽屉并关抽屉84%→76%；杯入微波炉并关门80%→70%；书入收纳架100%→98%。接口并未解决所有多阶段执行问题。
- Goal由62.2%到68.0%，净增加29次成功，其中cream cheese入碗由1/50到28/50，增加27次。开上抽屉放碗仍1/50；推盘子由5/50到0/50；开中抽屉17/50到18/50。后续若继续分析，需独立检查这些任务的位姿执行和阶段切换，不能只归因于夹爪。

## 全部子任务

### libero_object

| ID | 任务 | 旧成功率 | 本次成功率 | 变化（百分点） |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0% | 88% | +88 |
| 1 | pick up the cream cheese and place it in the basket | 2% | 100% | +98 |
| 2 | pick up the salad dressing and place it in the basket | 94% | 100% | +6 |
| 3 | pick up the bbq sauce and place it in the basket | 92% | 96% | +4 |
| 4 | pick up the ketchup and place it in the basket | 100% | 96% | -4 |
| 5 | pick up the tomato sauce and place it in the basket | 0% | 96% | +96 |
| 6 | pick up the butter and place it in the basket | 44% | 90% | +46 |
| 7 | pick up the milk and place it in the basket | 0% | 96% | +96 |
| 8 | pick up the chocolate pudding and place it in the basket | 0% | 98% | +98 |
| 9 | pick up the orange juice and place it in the basket | 0% | 100% | +100 |

### libero_10

| ID | 任务 | 旧成功率 | 本次成功率 | 变化（百分点） |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0% | 70% | +70 |
| 1 | put both the cream cheese box and the butter in the basket | 0% | 80% | +80 |
| 2 | turn on the stove and put the moka pot on it | 64% | 54% | -10 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 84% | 76% | -8 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 86% | 86% | +0 |
| 5 | pick up the book and place it in the back compartment of the caddy | 100% | 98% | -2 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0% | 66% | +66 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0% | 92% | +92 |
| 8 | put both moka pots on the stove | 34% | 38% | +4 |
| 9 | put the yellow and white mug in the microwave and close it | 80% | 70% | -10 |

### libero_goal

| ID | 任务 | 旧成功率 | 本次成功率 | 变化（百分点） |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 34% | 36% | +2 |
| 1 | put the bowl on the stove | 100% | 100% | +0 |
| 2 | put the wine bottle on top of the cabinet | 100% | 98% | -2 |
| 3 | open the top drawer and put the bowl inside | 2% | 2% | +0 |
| 4 | put the bowl on top of the cabinet | 90% | 98% | +8 |
| 5 | push the plate to the front of the stove | 10% | 0% | -10 |
| 6 | put the cream cheese in the bowl | 2% | 56% | +54 |
| 7 | turn on the stove | 96% | 100% | +4 |
| 8 | put the bowl on the plate | 98% | 100% | +2 |
| 9 | put the wine bottle on the rack | 90% | 90% | +0 |

## 运行与复核

双PPU、四仿真进程，启动器报告耗时23414秒，约6小时30分。completion.json已生成；后台启动器已经退出。逐条结果、视频和轨迹保存在结果目录。

汇总完成后独立复核1500条结果、对应视频文件与轨迹、worker日志和完整命令规则，复核结果单独写入final_result_audit.json。视频复核范围为存在性、非空与结果后缀，不代表逐一人工观看1500个视频。

旧数据来自10月19日至21日的threshold历史评测；本次采用每episode/chunk独立采样种子，部分Python/NumPy/JIT环境也与旧评测不同。两次评测不能视为只有夹爪规则不同的严格成对A/B。此限制不改变本次96%/73%/68%的实测成功率，但影响对全部增减的因果解释。

## 保存位置

结果目录：/mnt/workspace/Native_Policy/user/hjz/starVLA/playground/Evaluation/libero_object_goal_long_guarded_parallel_20261024

主要文件：summary.json、completion.json、historical_comparison.json、final_result_audit.json；子目录results/、traces/、videos/。

完整报告：docs/libero_guarded_full_eval_20261024.md。
