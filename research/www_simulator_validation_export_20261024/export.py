"""Export the existing GPU/simulator validation records as reviewable CSVs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "www_simulator_validation_export_20261024_r1"
WIDE = ARTIFACTS / "www_simulator_gpu_validation_pilot_analysis_20261019"
CROSS = ARTIFACTS / "www_cross_model_simulator_validation_20261022_r3"
MULTIWINDOW = ARTIFACTS / "www_simulator_gpu_validation_20261018_analysis_r1"
MULTIWINDOW_GPU = (
    ARTIFACTS / "www_simulator_gpu_validation_20261018_r2",
    ARTIFACTS / "www_simulator_gpu_validation_sarathi_20261018_r1",
)
CROSS_GPU = ARTIFACTS / "www_cross_model_20261021_r1"
RANK = ARTIFACTS / "www_policy_ranking_bisect_20261024_r1"
RANK_EARLIER = ARTIFACTS / "www_policy_ranking_fidelity_20261023_r1"

POLICIES = {
    "arrival": "SPARE",
    "srp": "SRP",
    "sarathi": "Sarathi",
    "native": "Default (Native)",
    "past_future": "Past-Future",
    "s3_proxy": "S3 proxy",
    "apt_serve_kv_rule": "APT-Serve KV rule",
    "uniboost_static": "UniBoost static",
    "sola_rule": "SOLA rule",
    "fixed256": "fixed256",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict], columns: list[str] | None = None) -> int:
    path = OUT / name
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def flag(value: str | bool) -> bool:
    return value is True or value == "True"


def ape(sim: str | float, gpu: str | float) -> float | str:
    if sim in (None, "") or gpu in (None, "") or float(gpu) == 0:
        return ""
    return 100 * abs(float(sim) - float(gpu)) / abs(float(gpu))


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group in sorted(groups.items()):
        ttft = [float(row["mean_ttft_ape_pct"]) for row in group
                if row["mean_ttft_ape_pct"] != ""]
        tbt = [float(row["p99_tbt_ape_pct"]) for row in group
               if row["p99_tbt_ape_pct"] != ""]
        matched = sum(row["agreement"] for row in group)
        output.append({
            **dict(zip(keys, group_key)),
            "pairs": len(group),
            "matched": matched,
            "agreement_pct": 100 * matched / len(group),
            "gpu_passed": sum(row["gpu_passed"] for row in group),
            "simulator_passed": sum(row["simulator_passed"] for row in group),
            "false_positive": sum(row["false_positive"] for row in group),
            "false_negative": sum(row["false_negative"] for row in group),
            "mean_ttft_mape_pct": mean(ttft) if ttft else "",
            "p99_tbt_mape_pct": mean(tbt) if tbt else "",
        })
    return output


def paired_rows() -> tuple[list[dict], list[dict]]:
    wide = []
    for source in read_csv(WIDE / "pairs.csv"):
        gpu, sim = flag(source["gpu_pass"]), flag(source["sim_pass"])
        kv = int(source["kv_blocks"])
        wide.append({
            "experiment": "qwen25_7b_wide_single_window",
            "model": "qwen25_7b",
            "sample": source["source"] + "_d1_w00",
            "policy": POLICIES[source["arm"]],
            "paper_kv_2048_7424": 2048 <= kv <= 7424,
            "agreement": gpu == sim,
            "false_positive": sim and not gpu,
            "false_negative": gpu and not sim,
            "mean_ttft_ape_pct": ape(source["sim_ttft_ms"], source["gpu_ttft_ms"]),
            "p99_tbt_ape_pct": ape(source["sim_tbt_ms"], source["gpu_tbt_ms"]),
            **source,
        })

    cross = []
    for source in read_csv(CROSS / "paired.csv"):
        gpu, sim = flag(source["gpu_passed"]), flag(source["simulator_passed"])
        cross.append({
            "experiment": "cross_model_four_heldout_windows",
            "policy": POLICIES[source["arm"]],
            "agreement": gpu == sim,
            "false_positive": sim and not gpu,
            "false_negative": gpu and not sim,
            **source,
        })
    return wide, cross


def paired_summary(wide: list[dict], cross: list[dict]) -> tuple[list[dict], list[dict]]:
    normalized = []
    for scope, batch in (
        ("qwen7b_all_672_7424", wide),
        ("qwen7b_paper_2048_7424", [r for r in wide if r["paper_kv_2048_7424"]]),
        ("cross_model_all", cross),
    ):
        for row in batch:
            is_wide = row["experiment"].startswith("qwen25")
            normalized.append({
                "scope": scope,
                "model": row["model"],
                "workload": row["source"],
                "kv_blocks": row["kv_blocks"],
                "policy": row["policy"],
                "gpu_passed": flag(row["gpu_pass"] if is_wide else row["gpu_passed"]),
                "simulator_passed": flag(row["sim_pass"] if is_wide else row["simulator_passed"]),
                "agreement": row["agreement"],
                "false_positive": row["false_positive"],
                "false_negative": row["false_negative"],
                "mean_ttft_ape_pct": row["mean_ttft_ape_pct"],
                "p99_tbt_ape_pct": row["p99_tbt_ape_pct"],
            })
    return (
        summarize(normalized, ("scope", "model", "policy")),
        summarize(normalized, ("scope", "model", "workload", "kv_blocks", "policy")),
    )


def capacity_data() -> tuple[list[dict], list[dict], list[dict]]:
    points = []
    for side in ("gpu", "sim"):
        for path in sorted((RANK / side / "search").glob("*.json")):
            data = json.loads(path.read_text())
            for point in data.get("points", {}).values():
                points.append({
                    "model": data["model"],
                    "workload": data["source"],
                    "kv_blocks": data["kv_blocks"],
                    "policy": POLICIES[data["arm"]],
                    "side": side,
                    "search_state": data["state"],
                    "nonmonotonic": data.get("nonmonotonic", ""),
                    "rate_qps": point["rate_qps"],
                    "passed": point.get("passed", ""),
                    "mean_ttft_ms": point.get("mean_ttft_ms", ""),
                    "p99_tbt_ms": point.get("p99_tbt_ms", ""),
                    "backlog_drift": point.get("backlog_drift", ""),
                    "source_record": str(path.relative_to(ROOT)),
                })
    points.sort(key=lambda r: (r["model"], r["workload"], int(r["kv_blocks"]),
                               r["policy"], r["side"], float(r["rate_qps"])))

    brackets = read_csv(RANK / "capacity_brackets.csv")
    grouped = defaultdict(dict)
    for row in brackets:
        grouped[(row["model"], row["source"], row["kv_blocks"], row["policy"])][row["side"]] = row
    paired = []
    for (model, workload, kv, policy), sides in sorted(grouped.items()):
        gpu, sim = sides.get("gpu", {}), sides.get("sim", {})
        complete = (gpu.get("state") == "complete" and sim.get("state") == "complete"
                    and bool(gpu.get("lower_qps")) and bool(sim.get("lower_qps")))
        paired.append({
            "model": model, "workload": workload, "kv_blocks": kv, "policy": policy,
            "gpu_state": gpu.get("state", ""), "simulator_state": sim.get("state", ""),
            "gpu_lower_qps": gpu.get("lower_qps", ""),
            "gpu_upper_qps": gpu.get("upper_qps", ""),
            "simulator_lower_qps": sim.get("lower_qps", ""),
            "simulator_upper_qps": sim.get("upper_qps", ""),
            "gpu_nonmonotonic": gpu.get("nonmonotonic", ""),
            "simulator_nonmonotonic": sim.get("nonmonotonic", ""),
            "gpu_tested_rates": gpu.get("tested_rates", ""),
            "simulator_tested_rates": sim.get("tested_rates", ""),
            "complete_pair": complete,
            "capacity_lower_ape_pct": ape(sim.get("lower_qps", ""), gpu.get("lower_qps", ""))
                                      if complete else "",
        })
    return points, brackets, paired


def run_rows(directories: tuple[Path, ...]) -> list[dict]:
    rows = []
    for directory in directories:
        for path in sorted(directory.glob("runs/*.json")):
            data = json.loads(path.read_text())
            row = {"source_record": str(path.relative_to(ROOT))}
            for key, value in data.items():
                if key == "metrics" and isinstance(value, dict):
                    row.update(value)
                elif isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    row[key] = value
            if "arm" in row:
                row["policy"] = POLICIES.get(row["arm"], row["arm"])
            if "model" not in row and directory in MULTIWINDOW_GPU:
                row["model"] = "qwen25_7b"
            row["side"] = "sim" if directory == RANK / "sim" else "gpu"
            rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = []

    def export(name: str, rows: list[dict], source: str, meaning: str) -> None:
        count = write_csv(name, rows)
        catalog.append({"file": name, "rows": count, "source": source, "meaning": meaning})

    wide, cross = paired_rows()
    export("01_qwen25_7b_wide_pairs.csv", wide, str((WIDE / "pairs.csv").relative_to(ROOT)),
           "All 29 KV levels and eight policies; paper_kv_2048_7424 selects the reported 22 levels.")
    export("02_cross_model_pairs.csv", cross, str((CROSS / "paired.csv").relative_to(ROOT)),
           "All three new models and all four measured policies, including fixed256.")
    export("03_qwen25_7b_four_window_points.csv", read_csv(MULTIWINDOW / "points.csv"),
           str((MULTIWINDOW / "points.csv").relative_to(ROOT)),
           "Separate four-window, two-repeat Qwen2.5-7B boundary confirmation batch.")
    export("04_qwen3_8b_ranking_pairs.csv", read_csv(RANK_EARLIER / "paired.csv"),
           str((RANK_EARLIER / "paired.csv").relative_to(ROOT)),
           "Earlier four-window Qwen3-8B ranking and fine-grid paired replays.")
    by_policy, by_kv = paired_summary(wide, cross)
    export("05_accuracy_by_model_policy.csv", by_policy, "01 and 02",
           "Agreement, false positives/negatives, TTFT and TBT MAPE for every model and policy.")
    export("06_accuracy_by_model_workload_kv_policy.csv", by_kv, "01 and 02",
           "Same metrics broken out by workload and KV budget.")

    points, brackets, paired = capacity_data()
    export("07_capacity_search_points.csv", points, str((RANK / "gpu/search").relative_to(ROOT))
           + " and " + str((RANK / "sim/search").relative_to(ROOT)),
           "Every locally recorded GPU and simulator QPS search point, including unfinished groups.")
    export("08_capacity_brackets.csv", brackets,
           str((RANK / "capacity_brackets.csv").relative_to(ROOT)),
           "GPU and simulator capacity intervals with complete/running/missing states.")
    export("09_capacity_gpu_sim_comparison.csv", paired, "08_capacity_brackets.csv",
           "One row per model/workload/KV/policy; error only for complete GPU/simulator pairs.")
    export("10_capacity_ranking_pairs.csv", read_csv(RANK / "ranking_pairs.csv"),
           str((RANK / "ranking_pairs.csv").relative_to(ROOT)),
           "Published ranking comparison snapshot; see 08 and 09 for incomplete groups.")
    export("11_wide_kv_capacity_brackets.csv", read_csv(WIDE / "capacity_brackets.csv"),
           str((WIDE / "capacity_brackets.csv").relative_to(ROOT)),
           "Earlier coarse-grid capacity intervals across the full Qwen2.5-7B KV scan.")
    export("12_wide_kv_spare_gains.csv", read_csv(WIDE / "capacity_gains.csv"),
           str((WIDE / "capacity_gains.csv").relative_to(ROOT)),
           "Earlier coarse-grid SPARE capacity comparisons; interval resolution is coarse.")
    export("13_qwen3_8b_fine_pooled_points.csv", read_csv(RANK_EARLIER / "pooled_points.csv"),
           str((RANK_EARLIER / "pooled_points.csv").relative_to(ROOT)),
           "Four-window pooled Qwen3-8B timing and pass/fail at each QPS.")
    export("14_qwen3_8b_fine_improvement.csv",
           read_csv(RANK_EARLIER / "pooled_improvement_prediction.csv"),
           str((RANK_EARLIER / "pooled_improvement_prediction.csv").relative_to(ROOT)),
           "Earlier Qwen3-8B SPARE-vs-baseline capacity gain predictions.")
    export("15_qwen25_7b_four_window_gpu_runs.csv", run_rows(MULTIWINDOW_GPU),
           "artifacts/www_simulator_gpu_validation_20261018_r2/runs and Sarathi supplement",
           "All 192 individual GPU replays behind the four-window aggregate table.")
    cross_gpu = run_rows((CROSS_GPU,))
    paired_keys = {(r["model"], r["sample"], int(r["kv_blocks"]), r["arm"],
                    float(r["rate_qps"])) for r in cross}
    for row in cross_gpu:
        row["formal_simulator_pair"] = (
            row.get("model"), row.get("sample"), int(row["kv_blocks"]),
            row.get("arm"), float(row["rate_qps"])) in paired_keys
    export("16_cross_model_gpu_runs_all.csv", cross_gpu,
           str((CROSS_GPU / "runs").relative_to(ROOT)),
           "All GPU scan/formal runs; formal_simulator_pair marks the 596 held-out comparisons.")
    export("17_capacity_window_runs.csv", run_rows((RANK / "gpu", RANK / "sim")),
           str((RANK / "gpu/runs").relative_to(ROOT)) + " and "
           + str((RANK / "sim/runs").relative_to(ROOT)),
           "All available per-window GPU and simulator runs behind the capacity searches.")
    write_csv("00_catalog.csv", catalog, ["file", "rows", "source", "meaning"])


if __name__ == "__main__":
    main()
