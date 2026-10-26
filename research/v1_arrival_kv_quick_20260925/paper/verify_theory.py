"""CPU-only consistency checks for the manuscript; no GPU or source mutation.

Run from any directory with python3 -B /path/to/verify_theory.py.
Numerical checks support, but do not replace, the written proofs.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from research.v1_broad12h_20260925.policy import ArrivalRate, formula, select
from research.v1_budget_20260922.policy import State


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected):
    check(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
          (actual, expected))


def main():
    rng = random.Random(20260925)
    count = 1000
    for _ in range(count):
        q = rng.uniform(128, 10000)
        free = rng.randrange(0, 4097)
        eta = rng.uniform(0.1, 20)
        a = rng.uniform(0.1, 10)
        d = rng.randrange(1, 100)
        p = q / (q + eta * free)
        # Separate severe-shortage model: X=D with probability p, else zero.
        # At unit rho and common cost a, the same formula is exact break-even.
        certificate_threshold = math.ceil(a / p)
        check(certificate_threshold * p - a >= -1e-12, 'Exact conditional certificate')
        if certificate_threshold > 1:
            check((certificate_threshold - 1) * p - a < 0, 'Minimal certificate threshold')
        exact = (free + q / eta) * math.expm1(eta * a / q)
        linear = a * (1 + eta * free / q)
        benefit = q / eta * math.log1p(eta * exact / (q + eta * free))
        close(benefit, a)
        z = eta * a / q
        check(-1e-12 <= exact / linear - 1 <= z * math.exp(z) / 2 + 1e-12,
              'Taylor threshold error')
        integer_benefit = sum(q / (q + eta * (free + k)) for k in range(1, d + 1))
        error_bound = eta * q * d * (d + 1) / (2 * (q + eta * free) ** 2)
        check(-1e-10 <= d * p - integer_benefit <= error_bound + 1e-10,
              'Integer tail error')

        # Direct expectation versus the exact integer tail-sum identity.
        masses = [rng.random() for _ in range(25)]
        total_mass = sum(masses)
        masses = [v / total_mass for v in masses]
        direct = sum(min(d, x) * prob for x, prob in enumerate(masses))
        tail = sum(sum(masses[k:]) for k in range(1, min(d, 24) + 1))
        close(direct, tail)

        work = rng.randrange(0, 100000)
        rate = rng.uniform(0, 100000)
        tau = rng.uniform(0.01, 3)
        block = 16
        h0 = 8 + 128 * free / (1 + work / block)
        ha = 8 + 128 * free / (1 + (work + tau * rate) / block)
        expected_difference = 128 * block * free * tau * rate / (
            (block + work) * (block + work + tau * rate))
        close(h0 - ha, expected_difference)
        check(math.ceil(ha) <= math.ceil(h0), 'Arrival threshold direction')
        if free > 0 and d > 8:
            trigger = block / tau * (128 * free / (d - 8) - 1 - work / block)
            check((d >= math.ceil(ha)) == (rate >= trigger), 'Arrival trigger')
        candidate_savings = [rng.randrange(1, 100) for _ in range(4)]
        n0, na = math.ceil(h0), math.ceil(ha)
        choose = lambda n: next((i for i, v in enumerate(candidate_savings) if v >= n), None)
        if not any(na <= v < n0 for v in candidate_savings):
            check(choose(na) == choose(n0), 'Quantization')
        maximum = max(candidate_savings)
        if ha > maximum:
            check(choose(na) is None, 'Inactivity')

    estimator = ArrivalRate(1)
    events = [(0, 512), (0.3, 200), (0.3, 300), (0.7, 4096)]
    for t, prompt in events:
        estimator.observe(t, prompt)
    at_one = estimator.at(1)
    close(at_one, sum(prompt * math.exp(-(1 - t)) for t, prompt in events))
    close(estimator.at(4), at_one * math.exp(-3))

    # A physically feasible state exhibiting the claimed 48-block switch.
    # Both prompts remain incomplete under the baseline and are not protected.
    rows = [State('p0', 0, 2040, 2040, 0, 0, False),
            State('p1', 0, 2040, 2040, 0, 0, False),
            State('r0', 0, 2041, 2040, 1, 0, False),
            State('r1', 0, 2041, 2040, 1, 0, False)]
    # The latter two represent waiting recomputation after an earlier output;
    # they consume the budget but are correctly excluded from W.
    args = dict(rows=rows, budget=1024, chunk=256, free=128, total=129,
                block_size=16, concurrency=32, max_model_len=8176,
                recent={'arrival_rate': 4096.0})
    basic = select(**args, threshold=formula(0))
    arrival = select(**args, threshold=formula(1))
    check((basic['threshold'], arrival['threshold']) == (72, 40), 'Example thresholds')
    check(not basic['changed'] and arrival['changed'], 'Example action switch')
    saving = arrival['base'].growth - arrival['selected'].growth
    check(saving == 48, 'Example physical saving')
    args['free'] = 4096
    args['total'] = 4097
    roomy = select(**args, threshold=formula(1))
    check(not roomy['changed'], 'Implementation inactivity')

    source = ROOT / 'artifacts/v1_arrival_kv_quick_20260925/comparisons.csv'
    with source.open(newline='') as stream:
        comparisons = list(csv.DictReader(stream))
    measured = [r for r in comparisons if r['family'] == 'measured_lengths']
    old = [r for r in measured if r['reference'] == 'old']
    check(len(old) == 56, 'Measured scenario count')
    with (source.parent / 'summary.json').open() as stream:
        summary = json.load(stream)
    check(summary['basic_runs'] == 708, 'Basic run count')
    check(summary['round_audit']['rounds'] == 1937134, 'Audit rounds')
    expected = {'old': -0.28, 'srp': 10.64, 'native': 40.52,
                'sarathi_common': 42.14, 'past_future_common': 26.89}
    overall = {}
    for reference, rounded in expected.items():
        selected = [r for r in measured if r['reference'] == reference]
        value = statistics.fmean(float(r['gain_pct']) for r in selected)
        check(round(value, 2) == rounded, ('Quoted comparison', reference))
        overall[reference] = value
    check(sum(float(r['gain_pct']) > 2 for r in old) == 6, 'Improved cells')
    check(sum(float(r['gain_pct']) < -2 for r in old) == 10, 'Regressed cells')
    capacities = []
    for capacity, expected_count, expected_zero in [(2560, 4, 4), (3072, 7, 4), (4096, 8, 7)]:
        subset = [r for r in old if int(r['capacity']) == capacity]
        zero = sum(float(r['arrival_interventions']) == 0
                   and float(r['reference_interventions']) == 0 for r in subset)
        check(len(subset) == expected_count and zero == expected_zero, 'Capacity counts')
        capacities.append(dict(capacity=capacity, cells=len(subset), zero_intervention_cells=zero,
                               gain_pct=statistics.fmean(float(r['gain_pct']) for r in subset)))
    examples = []
    for name, interventions, quoted_gain in [
        ('code_w2_burst_stop_c4096', 50, -3.668),
        ('conversation_w0_burst_stop_c3072', 12, -1.142),
    ]:
        row = next(r for r in old if r['group'] == name)
        check(float(row['arrival_interventions']) == interventions, 'Example intervention')
        for key in ['reference_interventions', 'arrival_preemptions', 'reference_preemptions',
                    'arrival_recomputed_tokens', 'reference_recomputed_tokens']:
            check(float(row[key]) == 0, ('Example zero field', key))
        check(round(float(row['gain_pct']), 3) == quoted_gain, 'Example gain')
        examples.append(row)

    # September 27 revision: verify the unchanged main policy and new evidence.
    from research.v1_arrival_kv_papers_20260925.common import ARMS as paper_arms
    from research.v1_mechanism_models_replicas_20260925.common import ARMS as model_arms
    from research.v1_replica_extension_20260925.common import ARMS as replica_arms
    for arms in (paper_arms, model_arms, replica_arms):
        check(arms['arrival']['config']['threshold'] == formula(1), 'A-only configuration')
        check(arms['old']['config']['threshold'] == formula(0), 'Basic configuration')

    def read_csv(path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    broad_path = ROOT / 'artifacts/v1_broad12h_20260925_r1/analysis/aggregates.csv'
    broad = [r for r in read_csv(broad_path)
             if r['family'] == 'measured_lengths' and r['scope'] == 'family']
    for arm, gain in [('arrival', 1.27), ('combined', -.65),
                      ('pressure_decode', 1.32), ('pressure_constant', .50)]:
        row = next(r for r in broad if r['arm'] == arm)
        check(int(row['cells']) == 160 and round(float(row['gain_old_pct']), 2) == gain,
              ('Broad formula comparison', arm))
    row = next(r for r in broad if r['arm'] == 'pressure_decode')
    check(round(float(row['gain_arrival_pct']), 4) == .0089, 'Direct pressure-D/A gain')
    reconciliation_path = source.parent / 'reconciliation.json'
    reconciliation = json.loads(reconciliation_path.read_text())
    check(round(reconciliation['new_four']['mean'], 2) == -.21, 'Four-repeat A ablation')

    full = ROOT / 'artifacts/qwen7b_full_metrics_20260926_r1'
    full_manifest = json.loads((full / 'manifest.json').read_text())
    check(full_manifest['scenarios'] == 30 and full_manifest['formal_global_episodes'] == 360,
          'New Qwen7B scope')
    full_comparisons = read_csv(full / 'comparisons_long.csv')
    paired = [r for r in full_comparisons if r['capacity'] == '2048'
              and r['replicas'] == '4' and r['case'] == 'steady' and r['load'] == 'per_replica']
    for reference, metric, gain in [
        ('old', 'ttft_mean', .13), ('srp', 'ttft_mean', 31.37),
        ('sarathi_common', 'ttft_mean', 64.52), ('sarathi_common', 'output_throughput', 20.93),
        ('past_future_common', 'ttft_mean', 34.67), ('past_future_common', 'e2e_p95', -2.10),
        ('past_future_common', 'output_throughput', -3.05),
    ]:
        row = next(r for r in paired if r['reference'] == reference and r['metric'] == metric)
        check(round(float(row['improvement_pct']), 2) == gain, ('New Qwen7B result', reference, metric))
    tail_counts = {}
    for metric, expected_positive in [('ttft_mean', 25), ('e2e_p95', 0), ('e2e_p99', 0)]:
        selected = [r for r in full_comparisons
                    if r['reference'] == 'past_future_common' and r['metric'] == metric]
        positive = sum(float(r['improvement_pct']) > 0 for r in selected)
        check(len(selected) == 30 and positive == expected_positive, ('Qwen7B directions', metric))
        tail_counts[metric] = positive

    extension_path = ROOT / 'artifacts/v1_replica_extension_20260925_r1/analysis/comparisons.csv'
    extension = read_csv(extension_path)
    for model, capacity, replicas, case, load, gain in [
        ('qwen7b', 768, 2, 'steady', 'fixed_total', 13.31),
        ('qwen7b', 768, 2, 'steady', 'per_replica', -14.87),
        ('smol360m', 2048, 1, 'burst_stop', 'fixed_total', -33.75),
        ('smol360m', 2048, 2, 'burst_stop', 'per_replica', -32.41),
        ('smol360m', 2048, 4, 'burst_stop', 'per_replica', -33.31),
    ]:
        row = next(r for r in extension if r['model'] == model and int(r['capacity']) == capacity
                   and int(r['replicas']) == replicas and r['case'] == 'conversation_w0_' + case
                   and r['load'] == load and r['reference'] == 'old')
        check(round(float(row['gain_pct']), 2) == gain, ('Cross-model ablation', model, replicas, load))

    probes_path = ROOT / 'artifacts/v1_mechanism_models_replicas_20260925_r1/analysis/probes.csv'
    probes = read_csv(probes_path)
    check(len(probes) == 6, 'Six selected branch states')
    for row in probes:
        area = statistics.fmean(float(row[k]) for k in ('gain_r0_area_ms', 'gain_r1_area_ms'))
        close(area, float(row['saved_waiting_area_ms']))
        close(area / 64, float(row['equivalent_mean_ttft_saved_ms']))

    # Basic structural and local-link checks on both text artifacts.
    import re
    for name in ['PAPER_DRAFT_CN.md', 'MANUSCRIPT_EN.md']:
        contents = (HERE / name).read_text()
        check(contents.count(r'\[') == contents.count(r'\]'), ('Display math', name))
        check(contents.count(r'\(') == contents.count(r'\)'), ('Inline math', name))
        if name == 'PAPER_DRAFT_CN.md':
            check(not any(delimiter in contents for delimiter in (r'\[', r'\]', r'\(', r'\)')),
                  'Chinese draft uses dollar math')
            check(contents.count('$$') % 2 == 0, 'Paired display dollar delimiters')
            without_display = re.sub(r'\$\$.*?\$\$', '', contents, flags=re.S)
            check(all(line.count('$') % 2 == 0 for line in without_display.splitlines()),
                  'Paired inline dollar delimiters')
            check([int(v) for v in re.findall(r'\\tag\{(\d+)\}', contents)] == list(range(1, 29)),
                  'Equation numbering 1 through 28')
            prose = re.sub(r'\$[^$]*\$', '', without_display)
        else:
            prose = re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)', '', contents, flags=re.S)
        for link in re.findall(r'\]\(([^)]+)\)', prose):
            if '://' not in link and link != 'verification.json':
                check((HERE / link).exists(), ('Missing local link', name, link))

    report = dict(
        passed=True, random_cases_per_identity=count, new_gpu_runs=0,
        numerical_checks=['exact severe-shortage certificate', 'finite-shortage root', 'Taylor relative error', 'integer-tail error',
                          'integer-tail identity', 'arrival threshold difference',
                          'arrival eligibility boundary', 'decision quantization', 'inactivity',
                          'EWMA event recursion and decay'],
        implementation_example=dict(basic_threshold=72, arrival_threshold=40,
                                    physical_saving_blocks=saving, roomy_state_inactive=True),
        overall_gain_pct=overall, capacity_observations=capacities, examples=examples,
        source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        revision_20260927=dict(main_policy='arrival = formula(1), no D coefficient',
                              qwen7b_scenarios=30, qwen7b_episodes=360,
                              past_future_positive_cells=tail_counts, mechanism_states=len(probes),
                              evidence_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in [broad_path, reconciliation_path,
                                                        full / 'comparisons_long.csv', extension_path, probes_path]}),
        caveat='Numerical checks do not prove theorems or validate conditional stochastic assumptions.')
    (HERE / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'random_cases_per_identity', 'new_gpu_runs',
                                             'implementation_example', 'capacity_observations']}, indent=2))


if __name__ == '__main__':
    main()
