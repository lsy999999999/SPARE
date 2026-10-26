# SPARE Evaluation 补齐协议

目标：在当前 SPARE 规则上完成缺失的模拟实验；主方法保持默认参数、排序、四候选及先达标选择。新目录隔离既有论文结果。

1. **版本与语义**：原生 vLLM 0.9.2 V1 CPU scheduler/allocator。默认 SPARE 必须与冻结 simulator 的完整轨迹哈希一致；覆盖有抢占的轨迹。输出长度仅执行器和明确标记的 oracle 可读。
2. **新输入**：Azure code/conversation 各三个预定日期窗口，每窗口取锚点后首512个合格、未见过的请求。合格条件为 prompt>0、0<output≤512、prompt+output≤8000，无长度裁剪。扫描历史 v1/www inputs/raw_inputs 排除重复；同一上游周轨迹不称新数据集。
3. **核心对照**：每窗口前128请求；steady=5.25 req/s、burst_stop=同时到达；配置KV块768/2048/4096；SPARE、basic、SRP、固定256/512预算、占用率70%时半预算、固定16/32省块门槛、去首token完成保护。所有简单规则事前给定，完整报告，不逐测试场景选择赢家。
4. **参数敏感性**：在768/2048块运行 n0=4/16、alpha=64/256、H=0.5/2秒、tau=0.5/2秒。每次只改变一项；默认8/128/1/1与tau=0复用核心对照。不是重选主方法参数。
5. **持续负载**：每窗口512个不同请求，2/5/10/20 req/s，三容量；SPARE/basic/SRP/固定512。记录到达四分位TTFT、排空时长、积压、TTFT/E2E/ITL、吞吐、抢占、重算。有限窗口不宣称最大可持续吞吐或长期稳定性。
6. **时间**：每条实验分别使用 resource、resource_staircase、gpu_fitted，不混合汇总。gpu_fitted为固定8特征非负最小二乘模型，只拟合已归档r3 fresh0窗口中arrival/SRP/native的前两次运行全部非空batch的engine.step墙钟；包含原插桩。冻结系数后用fresh1做逐轮和闭环逐请求验证。训练/验证ID不重叠，但这是历史数据回顾性划分，不是新采集profiling。结果始终保留performance_validated=false，报告误差与增益方向，不以拟合事实宣布验证通过。
7. **验证策略**：GPU r3 fresh1全部四重复平均与模拟配对，覆盖11个语义兼容的方法；TRAIL/UniBoost的CPU/GPU victim机制不同，排除该项误差比较，原13方法GPU主表不删。记录所有配对误差及方向；2%绝对GPU增益筛选仅作描述性计数。
8. **开销**：单进程CPU纯策略微基准，每个r3 arrival repeat0轨迹按轮索引均匀取32个状态，六次交替顺序测全候选审计与提前停止。核对动作一致，并检查去decode/finish保护的动作差异。排除排序/状态提取/原生调度/GPU执行，不称生产完整开销。
9. **重复与统计**：确定性模拟每格一次；不把重复同一轨迹当独立样本。相对TTFT降低以对照为分母，窗口内先平均，再平均六个来源/日期窗口。探索性区间按窗口成组重采样；容量、到达形状、参数、profile不是新样本。
10. **运行与失败**：12个CPU worker；每格最多500000轮/900秒。失败留档，整矩阵不中途按性能筛选；可恢复且锁定代码哈希。完成后自动生成完整CSV、报告和负载图。GPU实机校准/新窗口硬件复测不冒充由本次CPU模拟完成。

查看：`python3 -m research.www_evaluation_completion_20261003.status`

持续查看：`watch -n 10 python3 -m research.www_evaluation_completion_20261003.status`

恢复同一冻结批次：`.vllm-venv/bin/python -B -m research.www_evaluation_completion_20261003.run --jobs 12`

运行根目录：`.`。
