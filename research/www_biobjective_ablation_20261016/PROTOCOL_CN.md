# 同排序 mean TTFT–capacity 消融补测

沿用当前前沿图的样本、拟合时钟、预算、到达过程、QPS网格和通过判据。六档KV为768/1024/1536/2048/3072/4096；Code和Conversation各50个配对窗口，共72个「负载×KV×方法」组。其中12个完整SPARE组复用归档；60个消融组补测，每组逐QPS扫描直到首个失败，至少测到共同参考负载0.5 QPS。每个完成的单窗原子保存，可断点恢复。

## 变体与复用审计

|方法|唯一变化|目的|
|---|---|---|
|SPARE|完整规则|复用50窗口前沿图的数据|
|SRP|沿用同排序、关闭选择性缩减|整套推进控制的效果|
|Without arrival term|τ=0|近期输入项的贡献|
|Fixed threshold 16|n=16|与原消融一致的固定门槛|
|Fixed threshold 32|n=32|避免凭单个固定门槛作结论|
|Without first-token protection|关闭protect_finish，保留decode保护|本轮可完成prefill的保护贡献|

旧 `www_simulator_50rep_20261014_r1` 已包含这些消融，但使用另一批128/512请求流、到达模板及时间残差扰动，不能直接与当前Poisson capacity坐标拼接。本次复用它们的现有实现与参数，补测缺少的同条件结果。未搜索已有结果最有利的门槛，也不改现有SPARE算法。

复用源为 `artifacts/www_kv_capacity_50wide_20261016_r1`。启动前核对输入与时钟hash，并重放Code/768/0.5 QPS和Conversation/768/1 QPS各一个窗口，要求轨迹hash、延迟、积压、抢占和重算与归档完全一致。`smoke.json`记录检查结果。新变体逐一做有效运行检查；同一输入上某变体与SPARE轨迹相同也是合法结果。

## 测量协议

- CPU simulator，T=1024、K=256、并发32，块大小16 tokens，预留一个物理块；调度审计CPU开销未加入拟合时钟。
- 60秒Poisson输入、20秒预热；mean TTFT/P95 TTFT与P99 TBT基于20–60秒到达请求，追踪到全部完成。停止到达后的排空不增加capacity。
- 50个窗口全部有效，窗口P95 TTFT均值≤12秒、窗口P99 TBT均值≤200毫秒、40–60秒未完成请求数增长均值≤max(2,输入QPS)，该QPS通过。
- 12秒阈值是前沿实验的探索设置。本任务冻结该设置作同条件消融，不将其表述为新的独立SLO验证。
- 从0.25 QPS依次扫描原网格，报告首个失败前的最高通过点以及下一失败点。不内插，不将未扫描点称为实测，也不把有限窗通过等同长期稳定。
- 同负载同窗口的输入与随机种子配对。两种负载分别展示；图上不将两者平均冒充混合请求流容量。
- 完整回放的抢占/重算写入逐QPS表。它们包含排空阶段，应在共同QPS下比较，不能把各方法自己capacity处的记录当同负载因果比较。

## 输出与运行

启动：`sh research/www_biobjective_ablation_20261016/start.sh`。

进度：`sh research/www_biobjective_ablation_20261016/progress.sh`。

日志：`tail -f artifacts/www_biobjective_ablation_20261016_r1/controller.log`。

结束后自动生成 `analysis/points.csv`、`paired_deltas.csv`、`rate_metrics.csv`、`REPORT_CN.md`，以及分Code/Conversation的六面板PDF/SVG/PNG消融图。重叠点不抖动；虚线表示capacity通过到失败的离散扫描区间，不是置信区间。TTFT配对差值的bootstrap区间仅针对这些日期内的窗口，不代表日期总体不确定性。

进程异常时检查status.json及日志，解决原因后重用启动命令可续跑。controller.lock防止并行启动重复控制器。用户要求确认正常启动后即交接，不持续监控至实验完成。
