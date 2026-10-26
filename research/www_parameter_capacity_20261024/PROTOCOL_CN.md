**参数敏感性：同 KV 的 serving capacity 与 mean TTFT（2026-10-24）**

目的：将完整 SPARE 的单因素参数变化扩展至当前 Poisson 持续负载协议，在同一表中配对报告容量和同负载延迟。

- Workloads：Code、Conversation，各复用组件消融的 24 个冻结配对窗口；每类两来源日期、每日期 12 个窗口。
- KV：1024、2048、4096，容量与延迟完全使用相同档位、窗口和策略。
- 参数：默认 n0=8、α=128、H=τ=1 s，以及 n0=4/16、α=64/256、H=0.5/2 s、τ=0.5/2 s。一次只改一个参数，共九个配置。固定最终 n 不属于本矩阵。
- 执行：已有 GPU-fitted CPU simulator，T=1024、K=256、并发32；沿用同一输入采样、Poisson 随机种子、时钟和策略实现。
- 时间：60 s 到达，20 s 预热；统计 [20,60) s 到达请求并追踪到完成，积压差取 40–60 s。
- 容量：24 窗口全部有效，窗口 mean TTFT 的平均不超过 2 s 或4 s，窗口 P99 TBT 的平均不超过200 ms，平均积压增长不超过 max(2,QPS)。沿已测试 QPS 从低到高至首个失败，边界细化至0.01–0.02 QPS。
- 延迟：每个 workload/KV 下所有九配置共用同一 QPS。预先固定候选为2、1.5、1、0.5；选落在所有九配置2 s容量内、且九配置均满足上述服务约束的最高候选。优先2 QPS。表头注明最终QPS。
- 表格：行是参数配置，列按 KV 分组，每组并列 capacity(QPS) 和 mean TTFT(ms)。Code和Conversation分区排在同一张表。2 s为主表，4 s为对应补充表，延迟使用同一批测量。
- 复用：只复用冻结默认SPARE的同窗口原始记录；先重放三个归档轨迹，逐一核对哈希及延迟、TBT、积压，再启用缓存。八参数变体单独运行。
- 保留所有参数值与正负结果，不按最优参数更改主方法。扫描区间是数值分辨率，不是置信区间。

原始文件写入 `artifacts/www_parameter_capacity_20261024_r1/`。每个QPS的24条窗口记录可恢复；完整输入、代码、时钟hash写入design.json。运行器不修改旧实验文件。

执行：

```sh
.vllm-venv/bin/python -B -m research.www_parameter_capacity_20261024.run --smoke
.vllm-venv/bin/python -B -m research.www_parameter_capacity_20261024.run --run --workers 24
```
