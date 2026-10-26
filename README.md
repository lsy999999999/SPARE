# SPARE: Selective Prefill Scheduling

Anonymous research artifact for the 12-page manuscript
[`writing/www/www (32).pdf`](writing/www/www%20%2832%29.pdf). The code implements
SPARE on the vLLM V1 scheduler, the paired CPU simulator, external baseline
adapters, GPU validation, capacity scans, ablations, RL comparisons, and the
plotting and table scripts. The nearest editable LaTeX snapshot is in
[`writing/www/nowlatex/www(6)/`](writing/www/nowlatex/www%286%29/); the PDF is the
reference version for reported claims.

## Contents

| Paper item | Implementation | Frozen inputs and published aggregates |
| --- | --- | --- |
| SPARE and simulator | `research/www_evaluation_completion_20261003/`, `research/v1_budget_20260922/`, `research/v1_broad12h_20260925/`, `research/v1_saving_threshold_20260924/`, `simulator_core/` | `artifacts/www_evaluation_completion_20261003_r1/` |
| GPU baseline and simulator validation (Table 2) | `research/www_evaluation_gpu_completion_20261003/`, `research/www_simulator_gpu_validation_20261019/`, `research/www_cross_model_20261021/`, `research/www_simulator_validation_export_20261024/` | `artifacts/www_simulator_validation_export_20261024_r1/`, `artifacts/www_cross_model_simulator_validation_20261022_r3/` |
| Baselines, capacity and overall results (Figures 3-4, Table 3) | `research/www_baseline_fidelity_20261013/`, `research/www_paper_baselines_50rep_20261015/`, `research/www_kv_capacity_paper_20261015/`, `research/www_kv_capacity_fine50_20261016/`, `research/www_mean_ttft_capacity_20261017/` | Matching `artifacts/www_*` directories and `writing/www/experiments/gpu_baselines_4090_20260930_r3/analysis/`; see `analysis/*.csv`, `points.csv`, `means.csv`, `design.json` and input windows |
| Component ablations (Figure 5) | `research/www_component_ablation_mean_20261019/`, `research/www_biobjective_ablation_mean_20261019/`, `research/www_serving_capacity_ablation_20261021/` | `artifacts/www_component_ablation_mean_20261020_r4/analysis/`, `artifacts/www_biobjective_ablation_mean_20261019_r2/analysis/` |
| RL comparison (Figure 6) | `research/www_online_rl_20261017/`, `research/www_rl_standalone_v2_20261019/`, `research/www_rl_mixed_generalization_20261017/`, `train_rl_scheduler.py` | `artifacts/www_rl7047_frontier_20261020_r1/analysis/`, `artifacts/www_online_rl_20261017_r1/` |
| Parameter sensitivity (Table 4 and appendix) | `research/www_parameter_capacity_20261024/` | `writing/www/manuscript/sensitivity_capacity_20261024/combined.csv`, `matched_window_metrics.csv` |

The `research/*/PROTOCOL_CN.md` files describe the experiment settings and
commands in more detail. `writing/www/code/` and `writing/spare_mechanism_20261020/`
contain figure scripts. CSV and JSON aggregates are included so reported
numbers can be inspected without repeating a large GPU campaign.

## Environment and smoke test

Run from the repository root on Linux with Python 3.10. The CPU simulator
imports the vLLM 0.9.2 V1 scheduler and KV allocator but loads no model weights.
The original environment used vLLM 0.9.2, PyTorch 2.7.0, NumPy 2.2.6 and
SciPy 1.15.3. Install a vLLM 0.9.2 build suitable for your CUDA/PyTorch
installation, then install the plotting dependencies in `requirements.txt`.

```sh
python3.10 -m venv .venv
.venv/bin/pip install 'vllm==0.9.2'
.venv/bin/pip install -r requirements.txt
PYTHONPATH=.:plotting .venv/bin/python -m unittest \
  research.www_evaluation_completion_20261003.test_evaluation \
  research.v1_budget_20260922.test_scheduler \
  writing.www.code.test_simulator
```

For a small simulator replay, see
`research/www_evaluation_completion_20261003/simulate.py` and its CLI. Full
matrix commands are in each experiment's `PROTOCOL_CN.md`. Typical entry points:

```sh
PYTHONPATH=.:plotting .venv/bin/python -m research.www_paper_baselines_50rep_20261015.run status
PYTHONPATH=.:plotting .venv/bin/python -m research.www_parameter_capacity_20261024.run --smoke
```

GPU runs require an RTX 4090-class machine, the named model weights under
`models/`, and the corresponding input windows and vLLM installation. The
`MODEL` and `PYTHON` paths in GPU runners are repository-relative; launch them
from the repository root or set the experiment-specific environment variables.

## Data and scope

This release includes frozen length-only windows, calibration summaries,
experiment designs, paired aggregate CSVs, and the final manuscript figures.
The full Azure Code and Conversation one-week CSVs are not redistributed here.
Download the public traces from the [Azure LLM Inference Dataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
and place them at `data/azure_code_1week.csv` and
`data/azure_conv_1week.csv` for preparation or complete reruns that read the
source traces. Model weights are likewise obtained from their original model
publishers. The archived design hashes document the original experiment
snapshot; paths were made portable for this anonymous release, so compare
the included numerical inputs and outputs rather than expecting every
original source-file hash to match byte-for-byte.

The external baselines are rule-level adaptations or explicitly named proxies
on a shared V1 backend, as described in the paper. The included aggregate
tables allow the reported results to be audited; recreating every GPU and
million-run simulation measurement requires the original hardware, public
trace downloads, model weights, and substantial compute time.
