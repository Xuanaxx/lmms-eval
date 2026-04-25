import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pearson_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_var = sum(x * x for x in x_centered)
    y_var = sum(y * y for y in y_centered)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    covariance = sum(x * y for x, y in zip(x_centered, y_centered))
    return covariance / math.sqrt(x_var * y_var)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        for idx in order[start:end]:
            ranks[idx] = avg_rank
        start = end
    return ranks


def spearman_corr(xs, ys):
    return pearson_corr(rankdata(xs), rankdata(ys))


def residualize(target, control):
    if len(target) != len(control) or not target:
        return []
    control_mean = mean(control)
    target_mean = mean(target)
    control_centered = [value - control_mean for value in control]
    target_centered = [value - target_mean for value in target]
    control_var = sum(value * value for value in control_centered)
    if control_var <= 1e-12:
        return target_centered
    slope = sum(x * y for x, y in zip(control_centered, target_centered)) / control_var
    intercept = target_mean - slope * control_mean
    return [value - (intercept + slope * ctrl) for value, ctrl in zip(target, control)]


def summarize_run(name: str, results_path: Path, samples_path: Path, metrics_path: Path | None):
    results = json.load(results_path.open("r", encoding="utf-8"))
    samples = load_jsonl(samples_path)
    metric_rows = load_jsonl(metrics_path) if metrics_path and metrics_path.exists() else []
    metrics_by_doc = {int(row["doc_id"]): row for row in metric_rows if row.get("doc_id") is not None}
    metrics_by_sample = {int(row["sample_index"]): row for row in metric_rows if row.get("sample_index") is not None}

    merged = []
    for idx, row in enumerate(samples):
        doc_id = int(row["doc_id"])
        metric_row = metrics_by_doc.get(doc_id)
        if metric_row is None:
            metric_row = metrics_by_sample.get(idx)
        if metric_row is None and idx < len(metric_rows):
            metric_row = metric_rows[idx]
        merged.append(
            {
                "sample_index": idx,
                "doc_id": doc_id,
                "exact_match": float(row["exact_match"]),
                "adaptive_keep_ratio": (metric_row or {}).get("adaptive_keep_ratio"),
                "difficulty": (metric_row or {}).get("difficulty"),
                "attention_quality": (metric_row or {}).get("attention_quality"),
                "importance_entropy_norm": (metric_row or {}).get("importance_entropy_norm"),
                "importance_iqr_norm": (metric_row or {}).get("importance_iqr_norm"),
                "selected_iqr": (metric_row or {}).get("selected_iqr"),
                "importance_topk_mass": (metric_row or {}).get("importance_topk_mass"),
                "selected_scoring_layer": (metric_row or {}).get("selected_scoring_layer"),
            }
        )

    print(f"\n== {name} ==")
    print(f"overall_exact_match={results['results']['textvqa_val']['exact_match,none']:.4f}")
    if metric_rows:
        keep_ratios = [float(row["adaptive_keep_ratio"]) for row in metric_rows]
        difficulties = [float(row["difficulty"]) for row in metric_rows]
        print(f"avg_keep_ratio={mean(keep_ratios):.4f}")
        print(f"min_keep_ratio={min(keep_ratios):.4f}")
        print(f"max_keep_ratio={max(keep_ratios):.4f}")
        print(f"avg_difficulty={mean(difficulties):.4f}")

        layer_rows = [row for row in merged if row["selected_scoring_layer"] is not None]
        if layer_rows:
            log_layers = [math.log(float(row["selected_scoring_layer"])) for row in layer_rows]
            exact_matches = [float(row["exact_match"]) for row in layer_rows]
            importance_iqr_norm = [float(row["importance_iqr_norm"]) for row in layer_rows]
            print(
                "corr_exact_match_vs_log_selected_layer="
                f"{pearson_corr(log_layers, exact_matches):.4f}"
            )
            print(
                "rank_corr_exact_match_vs_log_selected_layer="
                f"{spearman_corr(log_layers, exact_matches):.4f}"
            )
            print(
                "corr_importance_iqr_norm_vs_log_selected_layer="
                f"{pearson_corr(log_layers, importance_iqr_norm):.4f}"
            )
            print(
                "partial_corr_exact_match_vs_log_selected_layer_given_iqr="
                f"{pearson_corr(residualize(log_layers, importance_iqr_norm), residualize(exact_matches, importance_iqr_norm)):.4f}"
            )

            layer_to_acc = {}
            for row in layer_rows:
                layer = int(row["selected_scoring_layer"])
                layer_to_acc.setdefault(layer, []).append(float(row["exact_match"]))
            for layer in sorted(layer_to_acc):
                layer_acc = layer_to_acc[layer]
                print(
                    f"layer_{layer}: n={len(layer_acc)} "
                    f"mean_exact_match={mean(layer_acc):.4f}"
                )

        merged.sort(key=lambda row: float("-inf") if row["difficulty"] is None else row["difficulty"])
        quartile_size = max(1, len(merged) // 4)
        for quartile_idx in range(4):
            start = quartile_idx * quartile_size
            end = len(merged) if quartile_idx == 3 else min(len(merged), (quartile_idx + 1) * quartile_size)
            chunk = merged[start:end]
            if not chunk:
                continue
            chunk_acc = mean(item["exact_match"] for item in chunk)
            chunk_keep = mean(
                item["adaptive_keep_ratio"] for item in chunk if item["adaptive_keep_ratio"] is not None
            )
            chunk_diff = mean(item["difficulty"] for item in chunk if item["difficulty"] is not None)
            print(
                f"quartile_{quartile_idx + 1}: "
                f"n={len(chunk)} acc={chunk_acc:.4f} keep={chunk_keep:.4f} difficulty={chunk_diff:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Summarize TextVQA adaptive keep-ratio experiments.")
    parser.add_argument("--run", action="append", nargs=4, metavar=("NAME", "RESULTS_JSON", "SAMPLES_JSONL", "METRICS_JSONL"))
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("Please pass at least one --run NAME RESULTS_JSON SAMPLES_JSONL METRICS_JSONL")

    for name, results_path, samples_path, metrics_path in args.run:
        summarize_run(
            name=name,
            results_path=Path(results_path),
            samples_path=Path(samples_path),
            metrics_path=Path(metrics_path),
        )


if __name__ == "__main__":
    main()
