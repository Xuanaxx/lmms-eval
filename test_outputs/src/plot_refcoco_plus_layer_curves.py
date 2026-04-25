import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RESULT_DIR = Path(
    "/home/user/czx/open_source_proj/lmms-eval/outputs/src_output/"
    "refcoco_plus_question_top64_layer_analysis_with_perf_corrected_bbox"
)


def load_layer_rows(summary_json: Path):
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = sorted(summary["layer_summaries"], key=lambda row: row["layer_idx"])
    return summary, rows


def metric(rows, name):
    return [row.get(name) for row in rows]


def save_layer_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_idx",
        "mean_importance_topk_bbox_recall",
        "mean_importance_topk_cell_iou",
        "mean_importance_topk_precision_overlap",
        "mean_importance_topk_precision_center",
        "mean_importance_auc_overlap",
        "mean_importance_score_mass_overlap",
        "mean_raw_attn_topk_bbox_recall",
        "mean_raw_attn_auc_overlap",
        "mean_dist_js",
        "mean_dist_kl_full_to_pruned",
        "mean_dist_total_variation",
        "mean_dist_logit_cosine",
        "dist_top1_match_rate",
        "dist_mean_top5_overlap",
        "performance_Bleu_4",
        "performance_METEOR",
        "performance_ROUGE_L",
        "performance_CIDEr",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def line_plot(ax, layers, values, label, marker="o"):
    ax.plot(layers, values, marker=marker, linewidth=1.8, markersize=4, label=label)


def annotate_best(ax, layers, values, mode="max"):
    if not values:
        return
    finite = [(layer, value) for layer, value in zip(layers, values) if value is not None]
    if not finite:
        return
    best_layer, best_value = max(finite, key=lambda item: item[1]) if mode == "max" else min(finite, key=lambda item: item[1])
    ax.scatter([best_layer], [best_value], s=64, zorder=5)
    ax.annotate(
        f"L{best_layer}: {best_value:.4f}",
        xy=(best_layer, best_value),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
    )


def plot_alignment(rows, output_path: Path):
    layers = metric(rows, "layer_idx")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    recall = metric(rows, "mean_importance_topk_bbox_recall")
    cell_iou = metric(rows, "mean_importance_topk_cell_iou")
    precision = metric(rows, "mean_importance_topk_precision_overlap")
    line_plot(ax, layers, recall, "Top64 bbox recall")
    line_plot(ax, layers, cell_iou, "Top64 cell IoU", marker="s")
    line_plot(ax, layers, precision, "Top64 overlap precision", marker="^")
    annotate_best(ax, layers, recall, mode="max")
    ax.set_ylabel("Overlap metric")
    ax.set_title("RefCOCO+ question input: top64 text-to-visual overlap with GT bbox")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    auc = metric(rows, "mean_importance_auc_overlap")
    mass = metric(rows, "mean_importance_score_mass_overlap")
    raw_recall = metric(rows, "mean_raw_attn_topk_bbox_recall")
    line_plot(ax, layers, auc, "Importance AUC overlap")
    line_plot(ax, layers, mass, "Importance mass inside bbox", marker="s")
    line_plot(ax, layers, raw_recall, "Raw-attn top64 bbox recall", marker="^")
    annotate_best(ax, layers, auc, mode="max")
    ax.set_xlabel("Scoring layer index")
    ax.set_ylabel("Alignment metric")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_distribution(rows, output_path: Path):
    layers = metric(rows, "layer_idx")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    js = metric(rows, "mean_dist_js")
    kl = metric(rows, "mean_dist_kl_full_to_pruned")
    tv = metric(rows, "mean_dist_total_variation")
    line_plot(ax, layers, js, "JS divergence")
    line_plot(ax, layers, kl, "KL(full || pruned)", marker="s")
    line_plot(ax, layers, tv, "Total variation", marker="^")
    annotate_best(ax, layers, js, mode="min")
    ax.set_ylabel("Distribution shift (lower is better)")
    ax.set_title("Full vs top64-pruned output distribution by scoring layer")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    cosine = metric(rows, "mean_dist_logit_cosine")
    top1 = metric(rows, "dist_top1_match_rate")
    top5 = [value / 5.0 if value is not None else None for value in metric(rows, "dist_mean_top5_overlap")]
    line_plot(ax, layers, cosine, "Logit cosine")
    line_plot(ax, layers, top1, "Top1 match rate", marker="s")
    line_plot(ax, layers, top5, "Top5 overlap / 5", marker="^")
    annotate_best(ax, layers, top1, mode="max")
    ax.set_xlabel("Scoring layer index")
    ax.set_ylabel("Similarity (higher is better)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_tradeoff(rows, output_path: Path):
    layers = metric(rows, "layer_idx")
    recall = metric(rows, "mean_importance_topk_bbox_recall")
    js = metric(rows, "mean_dist_js")
    auc = metric(rows, "mean_importance_auc_overlap")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax2 = ax.twinx()
    line_plot(ax, layers, recall, "Top64 bbox recall")
    line_plot(ax2, layers, js, "JS divergence", marker="s")
    annotate_best(ax, layers, recall, mode="max")
    annotate_best(ax2, layers, js, mode="min")
    ax.set_xlabel("Scoring layer index")
    ax.set_ylabel("Top64 bbox recall (higher is better)")
    ax2.set_ylabel("JS divergence (lower is better)")
    ax.set_title("Alignment vs distribution preservation")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    ax = axes[1]
    scatter = ax.scatter(recall, js, c=layers, cmap="viridis", s=54)
    for layer, x, y in zip(layers, recall, js):
        if layer in {6, 9, 18, 19, 21, 25}:
            ax.annotate(f"L{layer}", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Top64 bbox recall")
    ax.set_ylabel("JS divergence")
    ax.set_title("Layer trade-off scatter")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Layer index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_performance(rows, summary: dict, output_path: Path):
    layers = metric(rows, "layer_idx")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    cider = metric(rows, "performance_CIDEr")
    rouge = metric(rows, "performance_ROUGE_L")
    meteor = metric(rows, "performance_METEOR")
    bleu4 = metric(rows, "performance_Bleu_4")
    if any(value is not None for value in cider):
        line_plot(ax, layers, cider, "CIDEr")
        annotate_best(ax, layers, cider, mode="max")
    if any(value is not None for value in rouge):
        line_plot(ax, layers, rouge, "ROUGE_L", marker="s")
    if any(value is not None for value in meteor):
        line_plot(ax, layers, meteor, "METEOR", marker="^")
    if any(value is not None for value in bleu4):
        line_plot(ax, layers, bleu4, "Bleu_4", marker="D")
    ax.set_ylabel("Caption score")
    ax.set_title("RefCOCO+ caption performance after top64 visual-token pruning")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    recall = metric(rows, "mean_importance_topk_bbox_recall")
    js = metric(rows, "mean_dist_js")
    if any(value is not None for value in cider):
        line_plot(ax, layers, cider, "CIDEr")
    line_plot(ax, layers, recall, "Top64 bbox recall", marker="s")
    line_plot(ax, layers, js, "JS divergence", marker="^")
    baseline_cider = summary.get("baseline_performance_CIDEr")
    if baseline_cider is not None:
        ax.axhline(baseline_cider, color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.7, label="Full CIDEr")
    ax.set_xlabel("Scoring layer index")
    ax.set_ylabel("Mixed metrics")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RefCOCO+ layer-wise alignment and distribution curves.")
    parser.add_argument("--summary-json", default=str(DEFAULT_RESULT_DIR / "summary.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULT_DIR / "plots"))
    args = parser.parse_args()

    summary_json = Path(args.summary_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, rows = load_layer_rows(summary_json)
    save_layer_csv(rows, output_dir / "layer_metrics.csv")
    plot_alignment(rows, output_dir / "layer_bbox_alignment.png")
    plot_distribution(rows, output_dir / "layer_distribution_shift.png")
    plot_tradeoff(rows, output_dir / "layer_alignment_distribution_tradeoff.png")
    plot_performance(rows, summary, output_dir / "layer_pruned_performance.png")

    print(f"Saved plots and CSV to {output_dir}")


if __name__ == "__main__":
    main()
