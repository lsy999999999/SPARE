# Policy Ranking Fidelity 补充实验

模型为 Qwen3-8B，KV 配置为 2,048 blocks。Code 和 Conversation 各取四个此前未用于该模型计时拟合的窗口：`d1_w02`、`d1_w20`、`d2_w02`、`d2_w20`。比较 SPARE (`arrival`)、SRP 和 Sarathi；三种方法在同一负载及窗口使用完全相同的请求流和 QPS 网格。Code 网格为 0.5、1、1.25、1.5、1.75、2、2.25、2.5 QPS；Conversation 网格为 1、1.5、2、2.5、3、3.5、4、4.5、5、5.5、6 QPS。冻结设计见 `artifacts/www_policy_ranking_fidelity_20261023_r1/design.json`。

GPU 使用原有 vLLM V1 执行后端；模拟器采用此前拟合并冻结的 Qwen3-8B 逐轮计时参数。到达发生在 0–60 s，统计 20–60 s 到达请求的 mean TTFT 和 P99 TBT；积压增长取 40–60 s。达标要求为 mean TTFT 不超过 2 s、P99 TBT 不超过 200 ms、积压增长不超过 `max(2, QPS)`。在一个负载和 QPS 下，四个窗口的三项指标分别等权平均，再作联合达标判断，与正文容量扫描的聚合方向一致。

容量取从最低 QPS 起连续通过的区间 `[last passing QPS, first failing QPS)`。记录区间后的非单调通过点，不将其解释为更高的可靠服务容量。只有 GPU 和模拟器两边的容量区间都闭合，才比较方法排序。策略 A 严格领先 B 需要 A 的区间下界不小于 B 的区间上界；区间重叠记为未分辨，不计入严格排序一致率。另报告所有网格并列和未闭合组数，避免以并列抬高一致率。

若四窗口汇总的 Conversation 容量区间闭合且宽度为 0.5 QPS，在该策略的区间中点补测一次 GPU 与模拟器，并保留全部四个窗口；这是预定的一次二分细化。Code 的原始网格已为 0.25 QPS，不作这项细化。

对同时闭合的三个策略，以 SRP、Sarathi 的较大容量下界作为最强基线的网格估计；分别计算 GPU 和模拟器上 SPARE 的相对增益，两者相减为预测误差（百分点）。同时保留容量区间推出的增益上下界。旧批次相同输入的 52 个 GPU QPS 点使用当前 runner 重新回放，旧记录只作出处核查；最终排序使用单一 runner 版本。逐点结果、原始回放和逐轮记录保存在 `artifacts/www_policy_ranking_fidelity_20261023_r1/`。
