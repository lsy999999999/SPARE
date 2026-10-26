"""Freeze workload, models, comparisons and hardware-independent decisions."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_mechanism_models_replicas_20260925.common import *


def main():
    assert not (OUT / 'design.json').exists(), 'Use a fresh output directory'
    previous = ROOT / 'artifacts/v1_arrival_kv_papers_20260925_r1'
    cases = ['conversation_w0_steady', 'conversation_w0_burst_stop']
    for case in cases:
        data = read(previous / 'inputs' / (case + '.json'))
        assert len(data) == 64 and len({r['id'] for r in data}) == 64
        assert max(r['prompt'] + r['output'] for r in data) <= 8000
        write(OUT / 'inputs' / (case + '.json'), data)
    write(OUT / 'histories.json', read(previous / 'histories.json'))
    smol = Path('./.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/c38281e01d0c0b0c36eac2f5bcb5b51fa2e803fc')
    model_paths = {
        'qwen7b': Path('models/Qwen2.5-7B-Instruct'),
        'qwen05b': ROOT / 'models/Qwen2.5-0.5B-Instruct',
        'smol360m': smol,
    }
    models = {}
    for name, path in model_paths.items():
        config = read(path / 'config.json')
        assert config['max_position_embeddings'] >= 8176
        weights = sorted(path.glob('*.safetensors'))
        assert weights and all(p.stat().st_size > 1000000 for p in weights)
        head_dim = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
        bytes_per_token = 2 * config['num_hidden_layers'] * config.get('num_key_value_heads', config['num_attention_heads']) * head_dim * 2
        models[name] = dict(path=str(path), config=config, config_sha256=digest(path/'config.json'),
                            weights=[dict(name=p.name, size=p.stat().st_size, sha256=digest(p)) for p in weights],
                            fp16_kv_bytes_per_token=bytes_per_token)
    # Interleave model scale and memory regimes before looking at any outcome.
    tasks = []
    for capacity in (768, 4096, 2048):
        for model in ('smol360m', 'qwen05b', 'qwen7b'):
            tasks.append(dict(key=f'{model}_c{capacity}_r1', model=model, capacity=capacity,
                              replicas=1, loads=['fixed_total'], cases=cases))
        if capacity in (768, 4096):
            tasks.append(dict(key=f'qwen7b_c{capacity}_r2', model='qwen7b', capacity=capacity,
                              replicas=2, loads=['fixed_total', 'per_replica'], cases=cases))
    design = dict(frozen=True, created=time.time(), models=models, tasks=tasks,
                  cases=cases, capacities=[768, 2048, 4096], methods=list(ARMS), arms=ARMS,
                  repeats=2, max_hours=4, episode_timeout_s=300,
                  gpu_preference=[6, 7], max_model_len=8176,
                  planned_group_tasks=len(tasks), planned_global_episodes=312,
                  request_scope='Reused measured-length conversation w0; two arrival shapes; exploratory transfer, not independent confirmation.',
                  routing='Actual shared ingress, arrival-order round-robin; independent KV per replica; no oracle length routing.',
                  replica_scope='One GPU per replica. Fixed per-replica KV, budget and concurrency. Compare schedulers within each replica count; scaling alone is not scheduler benefit.',
                  model_scope='Same token-block quotas and length replay; KV bytes differ by architecture. No model-quality comparison.',
                  tuning='No parameter tuning; frozen basic+A. Same common config for paper-rule adaptations.',
                  histories_sha256=digest(OUT/'histories.json'),
                  trace_hashes={case:digest(OUT/'inputs'/f'{case}.json') for case in cases})
    write(OUT / 'design.json', design)
    print('Frozen', len(tasks), 'tasks; 312 global episodes; 408 per-replica episodes', flush=True)


if __name__ == '__main__':
    main()
