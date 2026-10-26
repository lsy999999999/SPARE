# WWW baseline RTX 4090 正式对比协议

目标是把 `writing/www/experiments/suite.json` 中 simulator 比较的全部13种方法放到真实 RTX 4090/vLLM 执行路径上测量。两个 simulator 时间模型共享相同 workload 和 KV 容量，在 GPU 上折叠为一个真实执行场景，因此正式设计为16个唯一场景、13方法、4重复，共832次正式 episode；每个 episode 固定64请求。

主方法为 `arrival`（论文稿中的 basic+A）。其余方法为 `basic`、`srp`、`native`、Sarathi、Past-Future、SSJF oracle、MC-SF oracle、TRAIL oracle、TIE oracle-proxy、UniBoost static、Apt-Serve KV rule、SOLA rule。所有方法使用同一 Qwen2.5-7B-Instruct、vLLM 0.9.2 V1、FP16、单卡、eager、无 prefix caching、无 speculative decoding、block size 16、`max_model_len=8176`、`max_num_seqs=64`、`max_num_batched_tokens=2048`、T=1024、K=256和场景指定的 KV block 容量。oracle/proxy 方法读取真实输出长度，不能作为可部署结果；rule/static 方法是论文规则在共享 vLLM 后端上的适配，不能写成论文原系统复现。

TRAIL 与 UniBoost 在本共享后端中只实现各自的优先级及 running/waiting 队列内排序。论文中的跨队列 victim replacement、KV swap 或有限抢占没有接入，因为 vLLM V1 的公开调度输出必须同步记录被抢占请求和 block table 变化；从扩展层直接释放 KV 会使 model runner 状态不一致。因此 GPU 结果不得表述为复现了这两种方法的抢占机制。

每个场景在同一张卡、同一个已预热 engine 中完成全部方法。四次重复采用两组预先确定的随机顺序及其反向顺序，降低时间漂移和顺序效应；不按中途性能调参、删方法或停止。每组正式测量前运行两次 `arrival` A/A，mean TTFT 与总时长的极差/均值均须不超过5%。组被外部GPU进程污染时整体排除并重跑，最多两次；不因结果好坏重跑。控制器硬截止为启动后24小时，未完成项不插值。

主指标为场景内四重复平均后的 mean TTFT，再对16场景等权汇总。次指标包括TTFT P95/P99、E2E mean/P95/P99、TPOT、max ITL、request/output throughput、TTFT<=5s且max ITL<=200ms的SLO达标率与goodput、KV峰值、抢占、重算token及调度CPU开销。相对结果使用16个场景的10,000次确定性bootstrap给出95%区间。

控制器只使用完全空闲且无compute PID的RTX 4090，并持有每卡文件锁；外部进程进入正在测量的卡会终止并隔离整组。所有原始逐请求时钟、逐轮调度轨迹、输入/历史/source SHA256、GPU UUID、驱动、CUDA、cuDNN、CPU、模型配置和软件版本均落盘。后台进程使用独立session、DEVNULL标准输入和文件日志，SSH断开不影响运行。
