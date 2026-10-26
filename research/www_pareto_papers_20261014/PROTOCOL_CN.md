# 外部论文方法的双指标 CPU 模拟

本轮按用户要求只讨论SPARE与外部论文／系统方法。SRP属于内部消融，不进入本轮主表或胜负统计。历史试验和原始结果保留。

## 冻结条件

复用上一轮140个输入：Code、Conversation各20个选参窗口及50个测试窗口；选参日和测试日分开；每窗口64请求，输出长度1–512、总长度≤8000，输入不截断。到达率5.25 req/s，KV容量768/4096块，每块16 token，1块保留。每个方法实现均扫描T=256/512/1024/2048、固定K=256和并发32。权重w=0,.1,...,1，按同输入native T1024归一化，先在选参集最小化J=w L/L0+(1-w)Q0/Q，再评价测试集。测试窗口每类来自两个日期，bootstrap按日期内窗口配对抽样。

CPU原生V1执行请求生命周期、分配、抢占和重算，耗时来自上一轮冻结的历史日志拟合公式；不加载模型、不运行GPU推理。SOLA增强版读取已有的独立合成输入校准参数，并接收本轮模拟耗时进行在线EMA更新。无需重新运行硬件校准。

模拟时钟不额外注入各方法的CPU选策耗时或缺失预测器的推理开销，调度耗时作为独立指标保存在原始结果。因此结果比较的是共同耗时模型下的调度轨迹，不包含完整系统的实现开销差异。

## 方法范围

主比较含SPARE及11个外部实现：vLLM V1、Sarathi、Past-Future原适配与多采样版、Orca预留、S³装箱代理、LTR分数代理、Apt-Serve KV规则、SOLA风险规则与校准版、UniBoost规则。这对应9个外部论文／系统方法族，两个版本不算两篇论文。

另测5个真值诊断实现：Proxy-SSJF、TRAIL、MC-SF、TIE、LTR oracle。合并主表与诊断表共覆盖13个外部方法族。完整列表和缺项见design.json的adapter_scopes；预测代理与oracle不能写成作者完整实现。

- Orca最大预留使用既有适配中的统一可见输出上限2048，实际测试输出≤512；保留共同chunk和V1分配，不复现selective batching。
- S³和LTR用冻结历史长度代理，缺少作者文本预测器。LTR的历史中位分数不能测试学习排序能力。
- UniBoost保留CPU版优先级victim replacement，固定gamma，无swap；Apt-Serve仅KV特例，无hidden cache。
- Past-Future-multi在小batch下采样32次取平均峰值；这些为明确的本地实现选择。SOLA-calibrated按5秒/200ms目标选phase和预算，接收真实模拟时钟反馈。
- 五个oracle诊断显式读取真实输出长度，只报告诊断结果，不混入可见信息主排名。
- TokenFlow需要客户端消费速率、buffer水位和迁移代价模型，现有长度trace缺少这些输入，因此本轮不报告它的数值。

## 运行与审计

17个实现（含SPARE）×4预算×2容量×140输入=19040条正式轨迹，选参5440条、测试13600条。每个测试配置50次不同窗口输入。全部配置预先固定，公开全部完成轨迹，不依据结果删除方法或选择性改变参数。多个版本在表中分开标注；“任何外部实现”的统计会将两版都纳入，不能误称每篇论文搜索预算只有4个配置。

接入验证先跑两类选参输入×17方法=34条CPU轨迹，检查全部完成与token/重算守恒；SPARE、Native、Sarathi、Past-Future与上一轮相同输入轨迹哈希完全一致；SOLA在线反馈确实更新。另运行既有多采样峰值、条件采样、非负拟合、EMA数学测试。正式运行继续验证完成数与吞吐分母，底层逐轮审计物理分配。源码和输入哈希前后核对。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_pareto_papers_20261014.run validate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_pareto_papers_20261014.run prepare
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_pareto_papers_20261014.run run --jobs 24
MPLBACKEND=Agg simulator/bin/python -B -m research.www_pareto_papers_20261014.analyze
```
