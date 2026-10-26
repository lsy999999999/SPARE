# WWW 中文实验稿配套 GPU 补充协议

2026-10-03。目的：验证简单控制的硬件效果，以及二维候选和精确块saving的作用。结果无论正负均保留，不据结果修改SPARE。

- 模型/引擎：Qwen2.5-7B-Instruct，RTX4090，vLLM0.9.2 V1，FP16，单GPU实例，eager，无prefix cache或speculation。B16、T1024/K256、策略并发32、context8176。重复token输入，执行器按给定输出长度终止；策略不读最终输出长度。
- 输入：复用本日CPU扩展中六个已冻结的新日期窗口，每个前128请求，steady=5.25req/s和burst-stop两形状；768/2048/4096配置块，每池一保留块。6×2×3=36场景。
- 14方法：arrival(SPARE)、basic(tau=0)、SRP、native、Sarathi规则、Past-Future规则、fixed256/512、constant16/32、no_finish_guard、budget_only、chunk_only、token_proxy。
- budget_only仅保留原前两个预算候选，K保持默认；chunk_only仅保留原第三个候选(T,min(64,K))，T保持默认。二者保留门槛、保护和fallback。
- token_proxy保持候选与物理可行性，仅以(base执行token总数−candidate执行token总数)/B替代精确新增物理块差来判断是否达到n。记录两种saving。该比值不向上取整、不改变物理分配器；这是一种明确的近似对照，不代表所有token近似算法。
- fixed256/512直接执行固定预算；constant16/32只替换门槛；去完成保护仍保留正常decode配额保护。
- 每场景4重复，2组预定随机/反向顺序，共2016正式运行；每组2次SPARE A/A，TTFT与总时长range/mean均≤5%。同组同卡、同预热引擎；不因性能胜负停止或重跑。
- 输入与源代码hash在运行前冻结。默认SPARE先与归档冻结实现核对模拟完整轨迹，全部变体在受限KV下完成原生调度检查、token/重算守恒及保护检查。
- GPU候选0/1/2/3/6/7，仅使用无compute PID、利用率0、显存占用<256MiB的空闲4090，沿用共享GPU文件锁；不结束其他任务。开始时7号卡空闲，其余卡空闲后可自动加入。
- 每episode超时900秒，单组watchdog12小时，全批72小时。外部GPU任务干扰整组隔离重试；最多2次重试，失败原始记录保留。未完成不插值、不从总体排名中无声删除。
- 指标：完整逐请求token时钟、TTFT/E2E/TPOT/maxITL、吞吐、SLO/goodput、KV、抢占、重算、调度开销、门槛/候选/保护/干预。先平均四重复，再按场景配对；探索性区间按六个源窗口成组bootstrap。
- 后台独立session、stdin=/dev/null、日志落盘。正常结束后自动核对hash并生成CSV/报告。脚本不需要本会话持续监控。

本批增加简单控制、因素消融和三个代表性外部对照的新窗口硬件结果。它不包含真实文本、新模型、完整作者预测器或完整系统重实现；这些范围在中文稿中仍明确列为未覆盖。
