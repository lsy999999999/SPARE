# SPARE 跨模型 GPU 试验

目标：冻结当前 SPARE 参数，检验模型计算速度和每 token KV 开销改变后，SPARE 相对 SRP、fixed256 和 Sarathi 规则的容量与 mean TTFT。旧版跨模型 GPU 实验只作背景，不与本批合并。

候选模型为 Qwen2.5-3B-Instruct、Qwen3-8B 与 Mistral-7B-Instruct-v0.3。分别检验规模与代际变化，以及不同模型系列；权重的 revision 或共享来源与 SHA256 记录在模型元数据。实验使用与 Qwen2.5-7B 验证相同的 4090、vLLM V1、FP16、固定输出长度、20–60 s 测量窗、40–60 s 积压增长窗、T=1024、K=256、并发上限 32、16 token/block。方法为 SPARE、SRP、fixed256、Sarathi 规则，SPARE 不调参。

先在 Code 与 Conversation 各自的 d1_w08 窗口，分别扫描每个模型和 KV 档的 QPS 边界；正式回放使用各负载的 d1_w02、d1_w20、d2_w02、d2_w20 四个窗口。Qwen2.5-3B 的预定 KV 档为 2048/4096/6144 块，Qwen3-8B 为 1536/1792/2048 块，Mistral-7B 为 1536/2048/2560 块；模型加载时若显存不允许，记录失败并调整到可运行的档位，不把调整后的结果与原设计混写。每块 16 token，实际 KV 字节量按模型架构记录；不同模型不按相同块数解释为相同显存压力。主指标为 mean TTFT <= 2000 ms、P99 TBT <= 200 ms、积压净增长 <= max(2,QPS) 下的最高通过 QPS，同时报告相同 QPS 的 mean TTFT、P99 TBT 和失败判据。所有负面结果保留。

每次回放保存确定性输入、原始请求时钟、逐轮调度记录、汇总判据及 SHA256。Qwen3-4B 的首轮只作环境预检，不进入跨模型结论。每组的正式 QPS 由预检扫描结果确定：各方法自身的通过/失败邻档，以及 SPARE 的最高通过档作为共同 QPS。正式任务在 GPU 回放前写入设计 JSON，不随结果修改。
