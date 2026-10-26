"""Independent synthetic calibration episodes on an idle RTX 4090, no test traces."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_legacy_baselines_20261006 import run as base
from research.www_baseline_fidelity_20261013.common import OUT, MODEL, write, digest


def main():
    gpu = int(sys.argv[1])
    folder = OUT / 'calibration'
    folder.mkdir(parents=True, exist_ok=True)
    lock = base.acquire(f'/tmp/query-capacity-gpu{gpu}.lock')
    hw = base.hardware()[gpu]
    if 'RTX 4090' not in hw['name'] or not base.idle(hw):
        raise RuntimeError('calibration GPU must be idle RTX 4090')
    write(folder/'start.json', dict(hardware=hw, started=time.time(), pid=os.getpid()))
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    engine = LLMEngine.from_engine_args(EngineArgs(model=str(MODEL), dtype='half',
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False, block_size=16,
        gpu_memory_utilization=.9, enforce_eager=True, disable_log_stats=True,
        seed=0, num_gpu_blocks_override=4096,
        scheduler_cls='research.www_evaluation_gpu_completion_20261003.scheduler.GPUBaselineScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    # Base runner loads only lengths, no dependency on old evaluation history.
    history = folder/'history.json'
    write(history, dict(lengths=[32]*1000))
    cell = dict(budget=1024, chunk=256, history_file=str(history))
    warm = [dict(id=f'w{i}', prompt=256, output=16, arrival_ms=0.) for i in range(4)]
    base.run_episode(engine,scheduler,warm,'native',cell,120)
    records=[]
    for repeat in range(3):
        for prompt,batch in [(p,b) for p in (64,256,1024,3072) for b in (1,4,8,16)]:
            external=base.compute_pids(hw['uuid'])-base.descendants(os.getpid())
            if external:
                raise RuntimeError(f'external GPU processes {external}')
            reqs=[dict(id=f'cal_{repeat}_{prompt}_{batch}_{i}', prompt=prompt,
                       output=24, arrival_ms=0.) for i in range(batch)]
            result=base.run_episode(engine,scheduler,reqs,'native',cell,120)
            if result['status']!='complete':
                raise RuntimeError(result['error'])
            records.append(dict(prompt=prompt,batch=batch,repeat=repeat,rounds=result['rounds']))
            write(folder/'profile.json', records)
    write(folder/'status.json',dict(state='complete',episodes=len(records),
          sha256=digest(folder/'profile.json'),updated=time.time()))
    lock.close()

if __name__=='__main__':
    main()
