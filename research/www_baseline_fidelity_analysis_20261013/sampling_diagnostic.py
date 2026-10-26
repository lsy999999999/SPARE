"""Sample-count sensitivity on history-only synthetic batches, no test outcomes."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *
from research.www_baseline_fidelity_20261013.policy import sampled_peaks,conditional_draws
results=[]
for source in ('code','conversation'):
    history=read(OUT/'history'/f'{source}.json')['lengths']
    for batch in (1,4,8,16):
        prompts=np.resize([128,512,1024,2048],batch)
        generated=np.resize([0,4,16,32],batch)
        rng=np.random.default_rng(20261013+batch)
        samples=conditional_draws(history,generated,8176-prompts,32000,rng)
        peaks=sampled_peaks(prompts,generated,samples)
        entry=dict(source=source,batch=batch,history_sha256=digest(OUT/'history'/f'{source}.json'),reference_mean_blocks=float(np.mean(peaks)),reference_sd_blocks=float(np.std(peaks)))
        for count in (1,8,32):
            estimates=peaks.reshape(-1,count).mean(axis=1)
            entry[f'samples{count}_estimator_sd_blocks']=float(np.std(estimates))
        results.append(entry)
write(OUT/'simulation/analysis/sampling_diagnostic.json',dict(protocol='synthetic prompt/progress batches; 32000 independent history draws; means of 1/8/32; no tuning or held-out workloads',results=results,source_sha256=digest(Path(__file__))))
print(json.dumps(results,indent=2))
