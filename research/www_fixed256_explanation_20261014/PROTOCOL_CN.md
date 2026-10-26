# Fixed-256 长输出解释实验

2026-10-14。设计、输入和源码哈希冻结于 `artifacts/www_fixed256_explanation_20261014_r1/design.json`；本地策略等价性、哈希检查和原生 V1 CPU 分配器回放均通过。4090 后台控制器已启动，正式运行等待候选卡空闲；状态与命令见[进度记录](PROGRESS_CN.md)。验证入口为 `research/www_fixed256_validation_20261014/validate.py`，结果位于同一 artifact 的 `validation.json`。

- 12 场景：Code/Conversation × 两个输出档各自最早的既有 64 请求窗口 × 768/2048/4096 块。
- 五版本：Full、Fixed-256、Accept-256、Recovery-256、No-finish-protection。每场景四次配对重复，共 240 次正式运行。
- 每场景两次 Full A/A；测量块前后各一次 Full。A/A TTFT 相对范围和前后 TTFT 相对漂移均不得超过 5%，否则保留失败尝试并重做整块，最多两次重试。
- 所有请求、到达时刻、输出停止长度、模型、后端、排序和容量配对固定；策略不可读真实停止长度。每次运行时限 21600 s，失败与超时不记作已完成延迟。
- 逐轮记录预算、chunk、实际调度 token、候选拒绝、空闲块、抢占、重算、调度时间和 step 时间。`step_ms - schedule_ms` 仅称非调度轮次耗时。
- 正式结果先按逐场景四次重复汇总。每个来源/输出档只有一个窗口，不据技术重复或容量派生场景构造总体窗口置信区间。

在 4090 主机的同一项目根目录同步本实验目录、验证入口及冻结 artifact 后，先用该机原有 vLLM 0.9.2 Python 环境重跑 `validate.py`，再执行 `run.py --launch`。控制器只使用空闲的 4090 GPU，并自动记录环境清单、运行状态、失败尝试和完成后的 `analysis/` 表格。待完成后，将结果按[解释方案](../../writing/www/manuscript/revision_v4_20261014/FIXED256_EXPLANATION_EXPERIMENT_CN.md)中的判据核对并更新正文。
