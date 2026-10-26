# 第 1 点：TTFT–输出吞吐量模拟验证

只执行 CPU 上的原生 vLLM V1 scheduler/block manager，不加载模型，不运行 GPU kernel。使用已冻结的历史 GPU 日志时间拟合公式推进模拟时钟；这仍是模拟结果。

预先固定五个方法族：SPARE、同排序不延后 SRP、native V1、Sarathi 规则适配、Past-Future 规则适配。各族扫描总预算 T=256/512/1024/2048，chunk K=256，并发上限32。SPARE 内部动态候选和门槛保持当前实现。各族都只有四次配置搜索；未覆盖各算法的全部参数空间。

Code、Conversation 各三个日期：第一个日期取20个选参窗口，另外两个日期各取25个测试窗口。每窗口64条原始请求，按时间顺序取连续合格请求，窗口间跳过64条合格请求；跳过每日期锚点后的前512条。输出长度在1–512、总长度不超过8000，不截断。到达间隔统一为1/5.25秒。Past-Future初始历史使用该来源第一个日期锚点后的512条已完成长度，与选参和测试请求不重叠。后续只更新已完成请求。请求身份也与时间模型拟合请求核对无交集。

每个输入运行两种KV容量768/4096块（每块16 token，一块为原生保留块）。共140个输入×2容量×5方法×4预算=5600条轨迹，其中测试4000条；每个测试配置有50个不同输入。五种方法共享输入和种子。六个来源/日期仍是六个日期簇，50窗口不能当作50个独立采集日期。

指标L为从计划到达到首token的平均TTFT；Q为全部实际输出token数÷从首个计划到达到最后完成的时长，包含排空，不计重算。此处Q是有限工作负载完成吞吐，不是最大可持续负载。

权重w=0,.1,...,1，评分J=w L/L0+(1-w)Q0/Q，参考固定为同输入native T1024/K256。先逐窗口归一化，再求平均。在选参集上对每个来源/容量/方法/w选择预算，平分取较小预算；测试集仅评价冻结的选择。同时公开所有配置测试点和经验非支配前沿。

95%区间按每个测试日期内部对25个窗口配对bootstrap，共5000次。日期权重固定，因此区间描述这两个日期内的窗口差异，不推断跨日期泛化。支配判断首先按测试均值；配对差异区间逐比较报告，为探索性、未作跨比较多重校正。

每条轨迹检查请求全部完成、输出token账目、吞吐计算；底层模拟器还检查KV归还、执行token与重算守恒。输入、时间公式、源码和哈希冻结于artifacts/www_pareto_20261014_r1/，所有运行保留，不按胜负挑选。

复现：

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_pareto_20261014.run prepare
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_pareto_20261014.run run --jobs 12
MPLBACKEND=Agg simulator/bin/python -B -m research.www_pareto_20261014.analyze
```
