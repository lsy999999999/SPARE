# 容量与论文规则小规模对比（结果前冻结）

目的：检查简单 1-block 在紧张 KV 下的优势及其在较宽松 KV 下的局限，
加入已复现的 Sarathi/Past-Future 规则。不开新策略、不做参数搜索。

两个已用开发窗口 code_test0、conversation_test0，原始到达间隔及输出长度；
容量 768、2048、4096 blocks。每窗口 64 请求，每 block 16 tokens。
这不是新窗口泛化检验，不据此声称容量决定最优动作。

八组：SRP(1024/256/32)、1-block、fixed512(512/256/32)、pressure_one(1/8)、
Sarathi-common、Past-Future-common、Sarathi-frozen、Past-Future-frozen。
三元组为总 token 预算／单请求 prefill chunk 上限／并发上限。
自适应组沿用 1024/256/32 基础计划与原候选列表；没有改成固定 chunk 的新算法。
common 使用共同 1024/256/32 配置，隔离排序/准入规则；frozen 使用上一轮
queue 自然输出场景已冻结的 1024/1024/32，保留论文风格的剩余预算 chunk。
不重新调参，不把旧 restricted/burst 参数当成当前窗口最优配置。

论文组是统一 vLLM V1 中的规则适配，绝非完整 Sarathi/LightLLM 原生系统。
沿用 research/v1_paper_comparison_20260922 的原实现及来源映射。
Past-Future 以源训练集 1000 个历史输出长度初始化，检查与比较请求 ID 不重叠；
只在实际完成后更新，条件采样种子 20260922+repeat，每段重置。
不读取评测请求真实未来输出，模型仅使用统一 context cap。

六个 cell，每个 cell 同卡执行八组×两次，首遍旋转方法顺序、第二遍反转。
共 96 次比较，每个 cell 另有固定 warmup 和两个 SRP A/A 校准，共 12 次校准。
A/A TTFT 及总时长 range/mean 均须 <=5%；超出则 cell 停止并保留全部记录。
GPU 使用空闲 RTX4090，优先 6/7/2，独占 advisory lock，固定不重叠物理 CPU 核。
三卡并行，每卡依次两个 cell；监测外部 CUDA PID，有干扰则整 cell 无效，
保留记录并报告，任何重跑必须记录原失败而不能挑选有利重复。

Qwen2.5-7B FP16，vLLM0.9.2 V1，同步 eager，无 prefix cache，context12288，
引擎上限 budget2048/concurrency64；实际每组由上述配置限制。每段限时300秒，
失败不计算部分请求 TTFT。记录每请求 token 时钟、每轮物理分配及源码/输入哈希。
首要指标 mean TTFT；E2E、p95 TBT 作为代价披露，不据其新增控制组件。
记录抢占、重算、干预、KV 占用及决策开销。公共审计开销包含在实际计时中。

CPU：统一分发器与原类在多轮紧张分配下逐动作、逐分配对照；另外六个 cell
八组共48段物理 KV 闭环模拟，所有请求同时到达、单位轮次，仅用于验证和机制，
绝不把模拟轮次当毫秒或用于替代真实 GPU 排名。

只报告探索证据。两次重复范围画为误差棒，不称置信区间；论文组配置分别报告，
不能从 common 的成绩推断论文最优实现表现。原有冻结代码和结果不修改。
