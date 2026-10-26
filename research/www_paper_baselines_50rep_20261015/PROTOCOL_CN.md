# SPARE 与外部论文 baseline：连续轨迹、每样本 50 次

## 目的与范围

纠正先前把内部 `basic`、`fixed256`、`SRP` 消融当作 0 号任务主比较的错误。本批主比较为 SPARE 和 11 个外部论文/系统规则及代理实现：vLLM V1、Sarathi、Past-Future、Past-Future multi、Orca reservation、S3 packing proxy、LTR score proxy、Apt-Serve KV rule、SOLA rule、SOLA calibrated、UniBoost rule。Past-Future 与 SOLA 的两个版本分别报告，不算两篇论文。另列 SSJF、TRAIL、MC-SF、TIE、LTR 五个真值长度 oracle/proxy 诊断；它们不进入可部署主排名。各适配器的具体保留机制和缺失部分写在 `design.json` 的 `adapter_scopes` 中，不能表述为作者完整系统复现。

复用已冻结的六个 Azure 来源日期，各取 10 段互不重叠的连续 512 请求窗口。主负载取每段前 128 请求，应用旧矩阵的 Steady/Burst-stop 到达模板；持续负载用全 512 请求，应用 2/5/10/20 req/s 模板。三档 KV 容量为 768/2048/4096 块；共同固定 T=1024、K=256、并发32，不在测试数据上调整预算。所有方法在相同「日期 × 样本 × 重复 × 到达场景 × 容量」下配对。每份样本每个实验单元运行 50 次，共 18000 个原子验收分片、918000 次 CPU 模拟。50 次的耗时扰动来自历史 4090 fresh0 训练轮的分阶段经验残差，同组方法用同一种子。单独的论文 Pareto 预算扫描见 `research/www_pareto_papers_20261014/`，本批固定配置不能替代那项调参实验。

Past-Future/S3/LTR 等所用历史输出长度，取每个来源首个日期锚点之前的最后 1000 个合格请求，与本批评估样本不重叠。原始轨迹保留连续请求的顺序、ID 与 prompt/output 长度；到达时刻沿用旧实验模板，不是 Azure 原始高并发时间戳。`design.json` 冻结样本、到达模板、历史、时间拟合、扰动池、适配器源码和本批源码 SHA256。旧内部批次与本批采用分别冻结的代码版本，不能拼接跨批绝对指标。

先在同一原始样本与扰动重复中计算 SPARE 对每个外部方法的 mean TTFT 和 output tokens/s 相对差异，再平均每个样本内的 50 次重复，最后报告 60 份样本的分布。六个来源日期仍是跨来源推断单位。五个 oracle/proxy 单独列出；CPU simulator 不覆盖真实模型、kernel、预测器及论文原系统的运行时开销。除主指标，还保存 E2E、TTFT P95、KV 使用、抢占、重算和调度 CPU 开销。

## 运行与续跑

本机项目根目录执行 `prepare` 后，按样本编号分区启动；已验收分片由 `.jsonl` 和含 SHA256、行数的 `.json` 共同确认，续跑仅补缺失或损坏的分片。不要并发运行重叠样本范围。远端仅缺原始 Azure CSV，`--host remote` 跳过原 CSV 检查，但继续核对冻结样本、历史、模板和全部源码哈希。当前分工为本机样本 0（1800 片）、远端样本 1–9（16200 片）。

```sh
.vllm-venv/bin/python -B -m research.www_paper_baselines_50rep_20261015.run prepare
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .vllm-venv/bin/python -B -m research.www_paper_baselines_50rep_20261015.run run --sample-start 0 --sample-stop 1 --jobs 24 --host local
```

远端使用隔离目录 `/path/to/remote/www_paper_baselines_50rep_20261015_remote` 和已安装的 `.venv/bin/python`，运行 `--sample-start 1 --sample-stop 10 --jobs 128 --host remote`。两侧完成后，只将远端 `artifacts/www_paper_baselines_50rep_20261015_r1/shards/` 同步到本机同名目录，再执行：

```sh
.vllm-venv/bin/python -B -m research.www_paper_baselines_50rep_20261015.run status
.vllm-venv/bin/python -B -m research.www_paper_baselines_50rep_20261015.run analyze
```

必须先验收 18000/18000 分片、918000/918000 次模拟，才能生成 `means.csv`、`paired_summary.csv`、`summary.json` 与 `REPORT_CN.md`。4090 Fixed512 任务保持暂停，按既有暂停文档恢复。
